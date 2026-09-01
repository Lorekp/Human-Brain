# Consequence-trained continual coding: established patterns and a neuronal path

Status: primary-source design note, 2026-09-01.
Scope: a replay-free continual coding learner that updates from authenticated
execution consequences, revises after failure, retains provenance and
confidence, and transfers repair structure. External claims below use original
papers, official project documentation/source, or official specifications.

## Executive conclusion

No surveyed system already provides the target combination.

- AlphaCode executes and behaviorally clusters many generated programs, but its
  test-time execution is selection over samples, not continual learning.
- AlphaDev genuinely learns from correctness and latency rewards, but inside a
  hand-built finite assembly game and through repeated training episodes.
- CodeRL turns compiler and test outcomes into typed critic targets and actor
  updates, but trains offline on generated programs and uses broad sampling at
  inference.
- Reflexion and modern coding agents close the action-observation-revision loop,
  but mainly improve by re-presenting textual history to a fixed language model.
- EWC and Synaptic Intelligence show replay-free synaptic stabilization;
  progressive networks show frozen reuse through lateral connections; neither
  supplies authenticated coding consequences or open-ended repair abstraction.
- Generative replay addresses forgetting by rehearsing pseudo-examples and is
  therefore outside this project's no-replay evidence constraint.

The reusable synthesis is: **a trusted executor emits a content-addressed,
typed consequence receipt; one immutable neuronal successor consumes that one
new receipt; local support, contradiction, and importance state update a
provenance-bound repair schema; low confidence abstains; a failed use inhibits
or specializes the schema; and only learned schemas, not prior examples, are
reused on a fresh task.**

This is the next seam after the roadmap's current iterator result. That result
authenticates selected source, compilation, execution, and a post-execution
oracle, but explicitly does not demonstrate live consequence learning,
debugging, or transfer ([roadmap](roadmap-reading-to-coding.md)).

## What the established systems actually establish

### Execution-filtered generation: AlphaCode

AlphaCode combines a large competitive-programming corpus, large
sampling-efficient transformers, large-scale sampling, and behavioral
filtering. Its paper reports that example tests remove more than 99% of samples;
the remaining programs are clustered by their behavior on generated inputs so a
small submission budget covers diverse behaviors. The system sampled up to one
million programs per problem in the reported analysis
([AlphaCode paper, Sections 4-5](https://storage.googleapis.com/deepmind-media/AlphaCode/competition_level_code_generation_with_alphacode.pdf)).

Reusable architectural pattern:

- treat execution as a semantic observation stronger than token likelihood;
- retain a behavior signature separately from source syntax;
- preserve diversity among hypotheses that behave differently;
- keep public/example tests distinct from hidden adjudication.

Benchmark/search technique, not a learning mechanism:

- drawing enormous independent samples and filtering them is test-time search;
- passing examples is not evidence of general correctness;
- behavioral clustering chooses among already-generated programs and does not
  revise a persistent learner after failure.

For this project, one authenticated execution should update memory; success
must not depend on million-sample search. Behavior signatures are useful as
physical consequence lanes and provenance, not as a host-side answer selector.

### Consequence-optimized program discovery: AlphaDev

AlphaDev casts assembly synthesis as a single-player game. Each action appends a
legal instruction. The reward combines program correctness with either program
length or measured latency; its network predicts a policy and value, while
Monte Carlo tree search selects actions and generated games train the network.
The system uses separate value heads for correctness and latency when optimizing
real latency
([AlphaDev paper](https://www.nature.com/articles/s41586-023-06004-9)).

Reusable architectural pattern:

- make the candidate program's real behavior part of the learning environment;
- keep correctness, performance, and other consequence types as separate
  channels rather than collapsing them prematurely into one pass bit;
- learn a prediction of future consequence so planning can prefer promising
  partial actions;
- validate final artifacts independently of the learner.

Domain-specific machinery:

- the assembly instruction vocabulary, CPU-state encoder, exhaustive small-sort
  inputs, reward weights, and microbenchmark setup encode this particular game;
- MCTS over a finite instruction game and repeated self-play are not evidence of
  replay-free continual repository learning.

The transferable lesson is typed consequence prediction, not the assembly game.
A neuronal coding learner should eventually predict compile, run, test, and
cost consequences separately and let physical competition integrate them.

### Compiler/test feedback learning: CodeRL

CodeRL models the code generator as an actor and trains a critic to predict
functional outcomes from generated programs. Its official implementation uses
four executor labels—compile error, runtime error, failed tests, and passed
tests—and stores the concrete error information and per-program test outcomes.
The critic provides denser training feedback; at inference, example tests and
critic scores drive program regeneration
([paper](https://arxiv.org/abs/2207.01780),
[official source documentation](https://github.com/salesforce/CodeRL#running-unit-tests)).

Reusable architectural pattern:

- preserve failure type and observation, not only a binary reward;
- learn both an action policy and a consequence predictor;
- make a failed execution causally change the next attempt;
- distinguish public feedback used for repair from hidden tests used for final
  evaluation. The paper itself reports a gap between example-test and
  hidden-test outcomes, warning that public examples admit false positives.

Benchmark/offline machinery:

- APPS-specific generated training programs, critic fine-tuning, and broad
  inference sampling do not constitute one-event continual learning;
- a learned critic score is not authenticated evidence and need not be
  calibrated outside its training distribution;
- regenerating many full programs can hide whether a reusable repair relation
  was learned.

For this project, the trusted executor's categorical outcome should drive
distinct physical input populations. A learned confidence circuit may predict
those outcomes, but only an authenticated receipt may update durable evidence.

### Failure reflection: Reflexion

Reflexion does not update the language model's weights. It converts scalar or
textual feedback into a verbal self-reflection and stores that reflection in an
episodic memory buffer for later trials
([paper](https://arxiv.org/abs/2303.11366)). In the official programming code, a
failed implementation is executed on tests, a reflection is produced from the
code and feedback, and the next generation receives the previous code,
feedback, and reflection
([official programming loop](https://github.com/noahshinn/reflexion/blob/main/programming_runs/reflexion_ucs.py)).

Reusable architectural pattern:

- separate raw consequence from a compact repair hypothesis;
- preserve the hypothesis across attempts and test it through action;
- revise the hypothesis when the next consequence contradicts it;
- keep a task-local loop bounded and terminate on verified success or explicit
  failure.

What must not be claimed:

- textual reflection in a prompt is context reuse by a fixed pretrained model,
  not synaptic continual learning;
- a fluent self-diagnosis is not evidence that the diagnosis caused the repair;
- internally generated tests are useful exploration tools but cannot serve as a
  protected oracle.

The neuronal analogue should store an executable, factorized repair hypothesis
and its consequence history, not a host-generated natural-language explanation.

## Continual-learning mechanisms under a strict no-replay constraint

### Elastic Weight Consolidation (EWC)

EWC slows changes to parameters important to earlier tasks by adding a
Fisher-weighted quadratic penalty around previously learned values. It does not
require old examples during the next task and demonstrated sequential retention
in classification and reinforcement-learning settings
([original PNAS paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC5380101/)).

Reusable pattern: maintain a second, slower state describing how costly it is to
change each learned connection. This is a stability-plasticity mechanism, not a
memory of old inputs.

Limits for this project: the original method assumes gradient training and
estimated Fisher importance around task transitions. Its importance only grows,
so capacity can eventually become too constrained; the paper discusses failure
after saturation. Repository work also lacks clean task boundaries. Therefore
copy the idea of consequence-derived plasticity control, not the Fisher loss.

### Synaptic Intelligence (SI)

Synaptic Intelligence accumulates an online, per-synapse estimate of each
parameter's contribution to loss reduction along the learning trajectory, then
uses it to protect important parameters during later learning. The authors
explicitly designed the importance estimate to be efficient and local to each
synapse
([original ICML paper](https://proceedings.mlr.press/v70/zenke17a.html)).

This is the closest established pattern to the simulation constraint. A local
neuronal implementation can maintain, per repair-schema synapse:

- successful causal contribution;
- contradictory contribution;
- consolidation/importance;
- remaining plasticity.

The project need not imitate SI's gradient formula. It can increase importance
only when independent authenticated consequences show that ablating or changing
the connection loses success. This keeps the update local and evidence-bound.

### Progressive networks

Progressive networks freeze the column learned for each prior task and add a new
column with lateral access to prior features. This prevents interference by
construction and permits feature transfer
([original paper](https://arxiv.org/pdf/1606.04671)).

Reusable pattern: immutable old capability plus a small learnable adapter is a
clean way to test whether prior structure transfers without mutating the source
capability.

Limits: capacity grows with tasks, and the original design requires the task
label to choose a column. A 100-billion-neuron target cannot afford one full
column per issue. Use a bounded bank of content-addressed repair schemas with
sparse lateral recruitment, not per-task copies. Context competition must choose
the schema; a host task ID must not do so.

### Deep Generative Replay

Deep Generative Replay trains a generator-solver pair to produce pseudo-inputs
and old targets, then interleaves those generated pairs with new-task data
([original NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/0efbe98067c6c73dba1250d2beaa81f9-Paper.pdf)).

This is a useful comparison but not an admissible implementation for the current
claim. Rehearsing generated substitutes is still replay: prior pseudo-examples
are actively reintroduced during the new update. It also makes retention depend
on generator fidelity. The broader architectural insight—fast particular memory
and slower compressed knowledge can coexist—remains useful, but consolidation
here must occur at first observation or through state dynamics that do not
re-present stored or generated lessons.

## Modern software-agent repair loops

### SWE-agent

SWE-agent's agent-computer interface defines both available commands and the
format of environment feedback. At each step the agent emits one thought and
command, executes it, then incorporates the observation. Its recorded
trajectories contain the response, parsed thought and action, observation,
environment state, and exact query; the configuration is saved for reruns
([paper](https://arxiv.org/pdf/2405.15793),
[trajectory documentation](https://github.com/SWE-agent/SWE-agent/blob/main/docs/usage/trajectories.md)).

Reusable architectural pattern:

- expose a narrow, reliable action protocol rather than raw unstructured host
  control;
- make each action produce one typed observation before the next decision;
- preserve the full trajectory for audit and debugging;
- keep interface errors separate from program consequences.

SWE-agent's trajectory-in-context loop is not itself continual learning. Under
this project's constraint, a trajectory may remain an external audit record,
but an update function must consume only the single new event and the immutable
parent state. It must not rescan the trajectory.

### OpenHands

OpenHands tracks a chronological event stream of actions and observations. The
agent maps event history to an action; the runtime maps that action to an
observation. Actions execute in an isolated runtime, and the official Docker
runtime uses a client-server boundary that returns formatted observations
([original platform paper](https://arxiv.org/pdf/2407.16741),
[official runtime architecture](https://docs.openhands.dev/openhands/usage/architecture/runtime),
[official current agent loop](https://docs.openhands.dev/sdk/arch/agent)).

Reusable architectural pattern:

- separate cognition, event custody, and the untrusted execution environment;
- model all tool effects as action-observation events;
- make steps atomic and interruptible;
- pin the runtime and dependencies so consequences are reproducible.

OpenHands is an execution and orchestration platform, not evidence that the
agent learns a durable repair schema. Its event-stream boundary is nevertheless
the right environmental seam for this project.

### Provenance as an execution receipt

SLSA build provenance binds an artifact to a builder identity, build definition,
external parameters, resolved dependencies, and output subjects so a consumer
can verify how it was produced
([SLSA provenance specification](https://github.com/slsa-framework/slsa/blob/main/spec/build-provenance.md)).
The direct analogue is a signed or otherwise root-authenticated consequence
receipt. Cryptographic roots establish custody; they must not be treated as the
neuronal representation of meaning.

## Architectural pattern versus benchmark trick

| Source | Reusable architecture | Technique that must not carry the claim |
| --- | --- | --- |
| AlphaCode | execution-derived behavior signature; public/hidden separation; hypothesis diversity | massive sample-and-filter search; example-test success as correctness |
| AlphaDev | action-consequence environment; separate correctness/cost predictions; independent final validation | handcrafted finite assembly game; exhaustive tiny domains; reward-specific MCTS |
| CodeRL | typed compile/runtime/test channels; learned consequence prediction; failure-conditioned retry | APPS critic fine-tuning; broad sampling; critic score as ground truth |
| Reflexion | compact failure-derived hypothesis tested on the next attempt | fixed-LLM prompt memory; fluent reflection as causal evidence; self-generated tests as oracle |
| EWC/SI | slow per-connection importance and plasticity state without example replay | global gradient/Fisher machinery; assumed task boundaries; permanent freezing at capacity |
| Progressive nets | immutable prior module plus learnable lateral reuse | one column per task; host-provided task identity; unbounded growth |
| Generative replay | distinction between particulars and compressed knowledge | any generated or stored-example rehearsal in a no-replay update |
| SWE-agent/OpenHands | typed atomic action-observation loop; sandbox; auditable event stream | repeatedly prompting a fixed model with history as evidence of learning |
| SLSA | authenticated builder, inputs, dependencies, environment, and outputs | hashes or attestations doing the learner's semantic decision |

## Proposed neuronal architecture

### 1. Keep three boundaries explicit

```text
environment adapter                 neuronal learner                 evidence store
-------------------                 ----------------                 --------------
tokenize / sandbox                  represent context                append receipts
compile / run / test    receipt ->  select repair schema    roots -> authenticate lineage
attest exact effects                predict consequences             never choose action
```

The environment adapter may tokenize, execute, measure, and authenticate. It may
not decide which repair is semantically appropriate. The evidence store may
verify custody. It may not rank candidates. Only simulated neuronal state may
select, revise, consolidate, bind, or abstain.

### 2. Define one authenticated consequence receipt

Each attempted action should yield one immutable receipt containing at least:

- parent learner checkpoint root and monotonically unique attempt identity;
- task/source root, repository tree before the action, and exact action/edit
  artifact root;
- repository tree after the action;
- executor identity, runtime/toolchain/dependency roots, command specification,
  resource limits, and test-manifest root;
- typed outcome lanes: interface/parse, compile, runtime, public tests, and
  optional cost/latency;
- raw observation roots plus bounded physical projections delivered to the
  learner;
- independence roots for repository, task source, test source, and execution;
- receipt root and authorization/custody chain.

This follows SLSA's builder/inputs/dependencies/outputs separation and the
OpenHands action-observation boundary. A result can be authentic and still be
wrong or incomplete; authentication answers *what happened where*, not *what it
means*.

### 3. Preserve a vector consequence, not one reward bit

Allocate sparse physical populations for at least:

```text
tool/interface valid
compile success / compiler-failure signature
run success / runtime-failure signature
public-test support and contradiction
resource/latency bucket
```

This adopts CodeRL's failure taxonomy and AlphaDev's separated correctness and
latency values. A later physical integration race may choose an action, but the
checkpoint must retain which channel caused support or inhibition.

### 4. Learn two coupled objects

1. **Consequence model:** context + proposed repair -> predicted typed outcome.
2. **Repair schema:** failure/context relation -> parameterized edit action.

The first enables planning and confidence; the second is reusable procedural
knowledge. Both should be sparse neuronal assemblies with local excitatory,
inhibitory, importance, and provenance-bound state. A repair schema should be a
generic token edit relation—such as insert/delete/replace/copy with learned
anchors and variable slots—not a Rust error name or task-family ID.

The current
[`common_continual_documentation_action_memory_v1.py`](../../experiments/r3_pilot/common_continual_documentation_action_memory_v1.py)
already supplies a useful base: immutable successors, replay rejection,
factor-action excitation for success, inhibition for failure, provenance rows,
promotion by independent task/source evidence, and physical abstention. Extend
that mechanism with live receipts and parameterized repair schemas; do not build
a parallel host policy.

### 5. Make revision an ordinary successor update

For one new receipt only:

- authenticate the parent and receipt;
- compare predicted and observed consequence through physical equality/error
  populations;
- on confirmed success, add local support and causal importance to the active
  context-schema connections;
- on failure, add typed inhibition to the active schema, preserve the failed
  hypothesis as provenance, and recruit or strengthen a competing specialization;
- if evidence is insufficient or conflicting, emit `STOP` or an experiment
  request rather than selecting by address/order;
- freeze and authenticate a new flat checkpoint; prove the parent is unchanged.

Do not replay old receipts, generated pseudo-receipts, or prior lessons. Existing
support/inhibition/importance arrays are state, not replay. An external audit
log may keep all receipts but the update API must not accept a batch or history.

### 6. Represent confidence as evidence, margin, and coverage

A single learned probability is too easy to mistake for truth. Confidence should
be a physically recomputed record with:

- independent positive and contradictory source counts;
- context coverage and unknown-factor count;
- predicted-consequence support by channel;
- positive-minus-inhibitory race margin;
- consolidated importance of the winning connections;
- exact roots of the evidence rows that contributed.

Promotion should retain the existing independent-source quorum idea. Confidence
must fall after an authenticated contradiction. A tie, inadequate coverage, or
dependent duplicate evidence must abstain. This turns metacognition into causal
state rather than a host-supplied score.

### 7. Transfer schemas, not source artifacts

Use the progressive-network pattern at repair-schema granularity: keep a
successful schema immutable, let a new sparse adapter bind it to a fresh context,
and permit a new branch when the old schema fails. Do not copy an entire network
per task. Context cells—not task IDs—must decide which old schemas are available.

A transfer claim requires all of the following:

- the schema was supported by at least two independent tasks and sources;
- the fresh task changes names, literals, surrounding syntax, and task family
  while preserving only the underlying repair relation;
- the learner binds fresh variables and emits a new edit artifact rather than
  recalling an old source hash;
- the action executes once under an authenticated public consequence surface;
- transfer improves success rate or executions-to-success versus an exact-memory
  control and a schema-off causal ablation;
- protected tests remain unopened until the action sequence is over.

## Smallest end-to-end progression

### C0 — Live negative consequence and revision

Reuse the current selected-source executor, but replace four pre-labeled
successful lessons with a stream containing a wrong action followed by a
successful alternative. Each compile/run/test receipt advances the same memory
exactly once. On recurrence, the failed action must be inhibited and the revised
action selected. This proves consequence-driven revision, not transfer.

### C1 — Two-source repair-schema consolidation

Provide two independent failed->successful repair episodes that share a generic
token-edit relation but not identifiers, literals, source text, or repository.
The network must discover the invariant edit anchors/slots, consolidate one
schema, preserve both provenance chains, and abstain after only one source. No
prior receipt may be replayed.

### C2 — Fresh schema transfer

Freeze the architecture and schema checkpoint before a third family-disjoint
task. The learner observes a fresh failure receipt, binds the learned edit
relation to new tokens, emits a novel source artifact, and receives one
authenticated consequence. Compare against schema-off, consequence-off,
provenance-quorum-off, and exact-artifact-only controls.

### C3 — Bounded repository repair loop

Place the same learner behind a SWE-agent/OpenHands-like narrow interface:
inspect, search, edit, compile/test, stop. Preserve every action-observation
receipt externally, but update neuronal state event by event. Add a hard action
budget and require recovery after at least one genuine failure. Only after this
works should larger SWE benchmarks be attempted.

## Evidence and anti-overfit rules

- Freeze architecture, tool protocol, tokenizer, consequence projections, and
  promotion thresholds before fresh transfer tasks.
- Learning may use public compiler/test consequences; protected adjudication may
  not feed back into the learner.
- Count executions, generated artifacts, and retries. Report any sample/filter
  search separately from learning.
- Require identifier and surface-form permutations, task-family disjointness,
  and at least one cross-language or non-coding structural falsification before a
  broad abstraction claim.
- Keep exact episodic evidence and compressed schema evidence separately
  ablatable. Removing the schema should remove the transfer advantage; removing
  exact evidence should not erase the already-consolidated invariant relation.
- Measure forgetting on old fresh probes after every new receipt. Use local
  synaptic importance only if its ablation causes verified retention loss.
- Treat provenance roots as custody and independent-evidence identities, never
  as semantic features that leak a task label or answer.

## Recommendation

Build C0 next. It is the smallest honest step from the authenticated iterator
cohort to closed-loop learning: one wrong selected action, one typed authenticated
failure, one replay-free successor update, one revised action, and one verified
success. Design its receipt so C1 can later replace the opaque action with a
generic parameterized edit schema. Add local importance only after interference
is observed; do not introduce EWC/SI machinery pre-emptively. This preserves a
working end-to-end product while opening the exact roadmap gaps of consequence
learning, revision, confidence, and eventually transfer.
