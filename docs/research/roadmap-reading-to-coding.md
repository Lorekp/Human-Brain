# Roadmap: a general learner for language, knowledge, and software

Status: living decision document. Updated 2026-09-06.
Owner: orchestrator. This document owns current priorities and architecture policy;
frozen experiment protocols retain authority over their own evidence.

## Objective

Build an intelligence network for real-world use that learns continually,
acquires and retains knowledge, discovers reusable abstractions, reasons and
plans, and improves through action and feedback. The ambition is eventually
superhuman capability; experiments must establish each step.

Language comprehension, memory, and software are the current focus. Coding
provides executable feedback; reading tests whether useful knowledge can be
acquired beyond a programming ontology. Vision and robotics remain deferred.

The long-run software targets are useful simple programming tasks, real
repository work, a fair SWE comparison exceeding Gemma-4-12B (quantized allowed),
then exceeding Opus 4.8 on DeepSWE. These are user-specified future targets:
verify exact model identities, availability, benchmark versions, and budgets
before preregistering a comparison. They are not present capability claims.

## Architecture policy

**Simulated-neuron learning remains the core research commitment for now.**
Cell dynamics, connectivity, plasticity, representations, and network growth may
change; GLIF and human-data fidelity are not mandatory global constraints.

**The main learner must use local, online cell/synapse learning**, as clarified
by the user on 2026-09-06. Updates may use a synapse's own state, pre/post activity,
locally maintained eligibility and explicitly declared teaching, reward or error
signals. Those signals must not conceal reverse-mode gradients through retained
network history or externally computed per-synapse updates. Backpropagation
through time, Adam and other global weight-fitting methods belong in separately
declared comparisons, not the main learner. Repeated experience and explicit
replay are allowed; count their exposure and cost. Locality is an architectural
constraint, not proof of biological fidelity or language understanding.
Vectorized simulation of these rules is allowed: locality concerns the information
used by an update, not whether the implementation uses a tensor operation.

The complete system need not be cell-only. Tokenization, conventional tooling,
external or hybrid knowledge stores, and components from other researchers are
allowed. Pretrained LLMs, including base/raw models as possible compressed
knowledge resources, are permitted research candidates. This is permission to
investigate, not a decision to integrate a particular model or store.

Reuse established work where useful. The research contribution should be a
learner that acquires and uses reusable structure, rather than an ordinary LLM
wrapper with an incidental neuronal component. For a hybrid proposal, state:

- what the neuronal core learns and controls;
- what the external component supplies, including any reasoning or action choices;
- how matched core-disabled, learning-disabled, and external-component-only
  controls will measure the core's contribution and improvement from experience;
- what weights, retrieved data, prompts, and compute enter the evaluated system.

Report hybrid-system and neuronal-component results separately. A final cell
race alone does not establish that the core learned the computation. Keep
target answers, hidden tests, and task-specific solvers out of undisclosed
scaffolding. Existing cell-only experiments keep their original contracts;
adding an external component requires a separately declared experiment.
External knowledge or features may supply inputs, but outsourcing understanding,
reasoning or action selection and then merely gating/copying the result does not
establish the neuronal core's capability. Any such assistance remains hybrid
evidence; it cannot establish neuron-only comprehension.
Mem0 remains development tooling outside the model and scientific pipeline.

## Immediate priority: one persistent end-to-end learner

R2, R3, and R4 are capability dimensions of one active workload, not independent
queues of mechanisms. The shared milestone is:

```text
read raw documents -> retain a fact after source removal
  -> use the fact to edit a small Python repository -> execute it
  -> learn from declared feedback -> transfer to a fresh task and retain prior skills
```

Use the new answer-free document learner in the actual action path. Keep the
first edit family narrow and explicitly label any demonstrated action schema.
Count executable behavior, not byte substitution in a text fixture, as software
success. Teacher corrections may follow first attempts; they do not count as
learner successes or autonomous recovery.

The first [narrow executed slice](../../experiments/r4_integrated/neutral_document_python_session.md)
remains positional-reading and literal-edit evidence. The adopted AROW/BINDING
baseline and earlier frozen results retain their contracts; see
[benchmark status](../benchmark-status.md) for their outcomes and limitations.
The [fixed-rehearsal comparison](../../experiments/r4_integrated/fixed_rehearsal_result.md)
establishes retention and execution of all sixteen taught questions when their
correct paragraphs are supplied. Conventional rehearsal also passes and fits sooner.

The completed [separate-answerability comparison](../../experiments/r4_integrated/answerability_pointer_result.md)
also fails qualification. Cells answer 25/192 supported development questions and
8/64 STOPs; BiLSTM reaches 23/192 and 3/64. Cells lose both initially correct old
actions during feedback; BiLSTM loses all nine. No updating policy gains fresh
supported success over its frozen mate. The saved-evidence audit passes. These
are exposed development results, not grounds for another automatic loss or
threshold sweep, architecture adoption or gate advancement.

At the user's request, [persistent memory, learned byte output and their action
integration](persistent-memory-byte-development.md) now advance in parallel.
These have independent gaps: the current reader needs supplied source positions,
while older assembly mechanisms depend on retained lexical fragments or supplied
live values. Develop the components
against shared interfaces and return them promptly to one persistent workload.
Their focused checks do not establish prose learning, source-absent transfer or
stable learning from action feedback.

The prepared BPTT/Adam joint fit was withdrawn before training when the user
specified local online learning. Its frozen recipe remains a possible comparison,
with no acquired-capability result. Reuse appropriate forward/state components
while developing local learning for the same persistent workload.

The [local byte-memory v5 result](../../experiments/r4_integrated/local_memory_byte_v5_result.md)
completes its first affordable local fit after reducing simulation overhead.
All updating arms emit one constant phrase across final questions, including
with writes disabled: useful source conditioning and learned-writing advantage
remain absent. Preserve the speed improvements, close this fit, and select one
bounded context-use/local-credit diagnostic from the saved failure distribution
and primary-source review before returning to the integrated workload. The
[direct-context control](../../experiments/r4_integrated/local_context_use_v1_result.md)
fits only 4/12 taught pairs and no withheld combination despite answer-revealing
inputs. The [symmetric-feedback comparison](../../experiments/r4_integrated/local_context_use_v2_result.md)
also fails: 3/12 taught and no withheld success versus random's 4/12 and zero.
Lower teacher loss and corrected EOS do not earn adoption. The
[learnability comparison](../../experiments/r4_integrated/local_context_learnability_v1_result.md)
now fits all twelve TRAIN pairs in the local phase/context reference, with
16 lexical or 15 permuted teacher mistakes, while the unchanged decoder reaches
only 5/12 under either order at 768 lessons. The reference transfers to only
2/4 or 3/4 withheld compositions, so neither architecture meets the useful
endpoint. Exact old-checkpoint parity and the reference certificate pass.
Retire further tuning of this decoder candidate. The
[composition review](local-composition-return-review.md) leads to a successful
[local potential-learning comparison](../../experiments/r4_integrated/local_context_lms_v1_result.md):
both training orders solve 12/12 TRAIN and 4/4 withheld combinations with genuine
EOS and minimum margins above 0.96; both controls score zero withheld. The
saved audit and a separate TRAIN-only represented-weight certificate pass.
Learning plus monitored audit takes 38.068 seconds. This earns the planned
limited two-byte return to actual document memory and executed actions, with
fresh output weights, explicit input-code ownership and upstream teaching.
Remove the authored factor inputs and requalify complete-workload cost; no
additional output diagnostic is queued. The supplied clock and fixed additive
representation remain priors, not learned reading or general output timing.
The [two-byte document return](../../experiments/r4_integrated/local_memory_two_byte_v1_result.md)
now completes its 5,120-lesson fit in 255.179 seconds after valid admission at
630.795 projected seconds; strict saved audit passes. Tagged, aggregate and
writer-frozen each solve 2/8 fit, 1/4 new-value, 1/4 delayed and zero interference
checks. Writes-disabled solves only 1/8 fit. Outputs respond to unrelated
document writes, but neither key/value assignment swap changes its answer.
Distinct retained matrices and changed slow weights therefore do not establish
useful binding or learned writing. Close this fit without more epochs or an
automatic local-rule tweak. The selected
[localization review](local-document-memory-return-review.md) directs one
60-second frozen-checkpoint diagnostic of actual writes, query addressing,
output potentials and isolated local update effects before another acquisition.
The [completed diagnostic](../../experiments/r4_integrated/local_memory_localization_v1_result.md)
and saved audit take 26.236 seconds. Assignment differences survive into reads
and output potentials. Positive writer and address updates reduce loss in all
eight cases per block, but change no answer. Source swaps produce largely
common-direction read changes for questions requiring opposite answer changes;
this supports a binding limitation without proving erasure or a wrong update
sign. The [follow-up review](local-memory-localization-followup.md) selects one
direct integrated comparison of instantaneous memory-read credit into the query
map and query cell. Keep writer learning, LMS output, rate and roster fixed;
compare the repaired five controls with a matched current-rule tagged reference.
Fresh six-arm cost admission is required before fitting. No additional
supplied-factor output diagnostic or protected gate is opened.
These byte learners use no pretrained encoder or LLM. The
[action-path review](local-memory-byte-preflight-review.md) identifies host-computed
matching in the legacy edit motor, however. Keep neuron-generated answer bytes
and that supplied action schema separate in claims. The exclusive-neuron goal
still requires learned neuronal action selection; a final physical race does
not establish it. Return useful acquisition to a direct learned action-output
interface before treating this as general software capability.

The closed acquisition diagnostics use explicit answer/STOP supervision over retained
raw sources and a frozen external encoder. These are not answer-free internalization,
fresh transfer, or a pass of the general integrated gate. Keep its controls and
costs explicit. Provenance revision, composition, full challenge and scaling
remain deferred until useful acquired and fresh behavior improves.

Alongside integration, apply two inexpensive sources of pressure:

- **Comprehension and transfer:** use authentic public-exam reading material and
  official marking guidance, not LLM-written evaluation questions. Follow the
  [reading evaluation plan](reading-comprehension-evaluation.md). Vary sentence
  length, question form, incidental lexical overlap, repeated entities, and
  distractors in public development. Renaming alone is not language generalization.
- **Growth cost:** measure useful accuracy, STOP/error rates, ingestion and query
  latency, total retained bytes, and work per new episode as facts, frame
  diversity, sentence length, and session length grow. Count retained derived
  evidence and repeated processing as well as raw data; zero retained documents
  does not imply bounded incremental work or semantic compression.

## Work-selection rule

Keep one shared integration target. Parallel agents may tackle independent
blockers, reference research, evaluation design, and measured performance costs.
Use roughly 3:1 effort on integrated capability versus isolated mechanism work.

Before a mechanism experiment, record its observed workload failure, reusable
hypothesis, comparison, completion criterion, and time/compute or iteration
budget. After its focused test, return to the shared workload. A mechanism pass
earns integration, not another automatic mechanism milestone.

If the budget expires without a useful gain, stop extending that mechanism,
summarize what was learned, and compare alternative explanations or architectures.
Repeated failures should trigger a bounded research/reasoning review. Preserve
negative results; replace ineffective approaches instead of accumulating
task-specific patches or defending sunk work.

Efficiency is integral to capability: faster useful simulation can support a
larger effective learner. Optimize measured end-to-end bottlenecks, preserving
the declared result. Kernel speedups that slow the workload are not improvements.
The suggested 100-billion-neuron scale is exploratory, not an acceptance target.
Distinguish addressable, stored, updated, and causally useful neurons; expand
scale when learning and resource curves justify it. Resource detail belongs in
the [scale study](100-billion-cell-architecture.md).

## Evidence and advancement

### Development versus adjudication

Public **TRAIN development** is repeatable: use regression suites, instrumented
failures, interface preflights, and reference implementations before freezing.
Keep broad strata that the current model cannot handle visible in the results.
A simplified single-match request may isolate a mechanism; it must not become
the universal language contract or erase realistic failures.

**Adjudication** uses a separately reserved fresh cohort and preregistered
architecture, scoring, controls, and resource budgets. Freeze only after ordinary
development contracts work. Spent adjudicative cohorts and artifacts remain
immutable; improvements require fresh evidence. These terms do not rename or
unseal the existing protected coding DEV/M0 surfaces.

For public exams, separate whole papers/years and shared passages according to
the reading plan, use official questions/rubrics, and record prior exposure.
Passage-visible comprehension and delayed source-absent memory are separate
tests. Public availability does not establish freedom from LLM pretraining
contamination, and an exam score alone is not general-intelligence evidence.

### What earns advancement

Require useful behavior change, not merely checkpoint mutation or physical
activity. Compare an updating learner with a frozen learner from the same
initial checkpoint; where stochastic, use matched seeds and multiple fresh
development streams. Include simple memorization/template/retrieval references
appropriate to the workload, plus causal ablations. Keep reference algorithms
outside the evaluated learner unless explicitly testing a hybrid architecture.

Report first-attempt executable success and reading marks, later-task learning
gain, recovery, retention/forgetting, sample efficiency, and total resource cost.
Separate correct abstentions from solved tasks and unsuccessful retries from
recovery. Inspect failure distributions and gains within each capability rather
than hiding a weak lane behind an aggregate score.

Before advancing beyond the integrated development gate, one persistent learner
must improve later fresh tasks over the frozen control, retain prior skills,
use acquired knowledge in real actions, and recover from some failures through
its own multi-step behavior. Numerical margins, uncertainty estimates, and budgets
belong in the successor protocol and must be fixed before its adjudication.
The initial narrow integration slice is progress toward this gate, not its pass.

Distinguish replay, composition, within-domain transfer, cross-domain learning,
and positive cross-domain transfer. Treat reusable compression, source revision,
uncertainty-driven evidence seeking, and longer plans as subsequent workload
requirements. Self-models and internal drives come later, when sustained agency
provides something meaningful to model and regulate.

## Stage and evaluation order

| Stage | Advancement requirement |
| --- | --- |
| R1 — substrate, continuous | Auditable learning and inference, explicit state mutation, credible resource accounting. |
| R2–R4 — memory, reasoning, language/software, active | Small integrated session, then broader comprehension, revision, composition and autonomous correction; satisfy the integrated development gate above. |
| G0 — cross-domain gate | Freeze the common architecture and learning rules; pass fresh adjudication in two non-coding domains under the [G0 protocol](frozen-architecture-cross-domain-generalization.md). |
| R5 — protected coding transfer, sealed | Integrated development gate and G0 pass on the architecture to be evaluated, then one protected family-disjoint coding evaluation. |
| R6 — prompt-only Simple-SWE, sealed | R5 pass and the existing frozen M0–M3 prerequisites. |
| R7+ — real-world software and comparisons | Increasingly difficult repositories and continual work; preregister fair Gemma and later Opus/DeepSWE comparisons. |

Public-exam development and its separately reserved reading assessments can
inform R2–R4; they neither replace G0 nor unlock protected coding evaluation.
A G0 STOP does not authorize R5. Material changes after a gate require fresh
validation of the affected claims; a prior architecture's pass does not transfer
automatically. Existing numerical protocols and spent evidence are unchanged.

## Where detail belongs

- [Benchmark status](../benchmark-status.md): measured results and limitations.
- [Program map](../program-map.md): implementation inventory, not a competing queue.
- [Experiment history](roadmap-experiment-history.md): chronology and retired paths.
- [Architecture](../architecture.md): current implementations and scoped contracts.
- [Reusable goal prompt](../project-goal.md): execution instructions for future work.

Read this roadmap before choosing work and after each context compaction. Load
the linked plan for the active milestone; old implementation plans are evidence
and design inputs, not authority to override current priorities.
