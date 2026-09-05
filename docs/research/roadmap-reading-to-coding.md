# Roadmap: a general learner for language, knowledge, and software

Status: living decision document. Updated 2026-09-05.
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

The [fresh-source failure](../../experiments/r4_integrated/fresh_source_result.md)
led to the now-completed
[token-pointer comparison](../../experiments/r4_integrated/token_pointer_result.md).
It also fails qualification: cells answer 18/192 untaught supported questions and
26/64 STOPs; the conventional reader reaches 23/192 and 28/64, missing its STOP
floor too. Cells retain initially correct old actions and improve old supported
execution 5/12 to 7/12, but gain no fresh supported success over their frozen mate
and lose one fresh STOP. The full saved-evidence audit passes. No architecture
is adopted or integrated gate passed.

The completed [decision review](token-pointer-followup-review.md) selects one
[separate-answerability development comparison](answerability-pointer-development.md).
It keeps producers, data and the action return fixed while testing separate
answerability credit and supported-only span learning. Source routing alone
cannot repair fresh comprehension; fitted old answers and declining loss do not
establish reusable acquisition. The reused probe is exposed development evidence.

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
