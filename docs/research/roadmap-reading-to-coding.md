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
network history or externally computed per-synapse updates. The user reaffirmed
**no backpropagation** on 2026-09-06: do not start backpropagation/BPTT or
Adam/global-fitting runs for this goal. Earlier frozen comparisons remain
historical evidence, not authorization for new backpropagation runs. Repeated experience and explicit
replay are allowed; count their exposure and cost. Locality is an architectural
constraint, not proof of biological fidelity or language understanding.
Vectorized simulation of these rules is allowed: locality concerns the information
used by an update, not whether the implementation uses a tensor operation.
This prohibition includes manually coded reverse-Jacobian error transport through
fixed forward maps. A local final synapse product or absence of autograd does not
make such an upstream error generator compliant. A synapse's direct output-neuron
delta rule remains distinct from propagating that error backward through another
forward stage.

**Learning and execution must use the same neuronal system**, also reaffirmed
by the user on 2026-09-06. The network that chooses an output/action must retain
and locally update its own learned state from subsequent feedback. Do not train
a separate model and transfer its solution into a nominal neuronal executor.
Teacher-driven activity and free action can differ within that same circuit;
declare those paths and their shared synapses explicitly. Frozen checkpoints
and source-free workers are diagnostic/custody tools, not a substitute for the
persistent online learning-and-acting loop. Conventional tool execution does
not establish learned action selection when a host solver chooses the action.

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
specified local online learning. Its frozen recipe remains a historical archive,
with no acquired-capability result or authorization for gradient training.
Reuse appropriate forward/state components
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
The [completed query-credit comparison](../../experiments/r4_integrated/local_memory_query_credit_v1_result.md)
passes admission at 853.567 projected seconds and finishes with its saved audit
in 369.946 seconds. Tagged improves fit from 2/8 to 5/8, but new-value and delayed
success remain 1/4 and interference remains zero. Writer-frozen matches tagged
correctness on all twenty checks; tagged still ignores both assignment swaps.
The old-rule arm exactly reproduces its prior trajectory. Close this fit with a
scoped taught-task gain, without useful transfer qualification or an automatic
epoch/rule sweep. The completed [writer-credit review](local-memory-writer-credit-review.md),
supported by [primary sources](local-memory-writer-credit-sources.md), selects
one direct-value-credit comparison. Existing tagged value traces already encode
the conditional effect of value synapses on retained memory; the corresponding
read-neuron error completes that path. Keep controller/key/strength approximations,
the query repair, rate and roster fixed, and compare with the immediate predecessor.
Ordinary arithmetic/control parity QA and fresh six-arm cost admission precede
any fit. The [completed value-credit comparison](../../experiments/r4_integrated/local_memory_value_credit_v1_result.md)
passes its fresh 850.126-second cost projection and completes with independent
audit in 364.919 seconds. Tagged remains at 5/8 fit, 1/4 new-value, 1/4 delayed
and zero interference success, matching its immediate predecessor and
writer-frozen control. Aggregate falls to 4/8 fit. The reference reproduces its
prior trajectory exactly. Conditional value-credit correctness has not produced
useful binding or a learned-writing advantage. Close this fit without more
epochs or another automatic error-broadcast change. The next bounded
[binding review](local-memory-binding-review.md) asks whether record identity
survives until the value arrives and influences the write address. Use saved
evidence first, then one declared frozen-checkpoint localization if necessary;
no further acquisition or protected gate is opened by this negative result.
The [completed binding diagnostic](../../experiments/r4_integrated/local_memory_binding_v1_result.md)
and independent audit take 16.405 seconds. Current identifier-related activity
contrast retains about 10% by value arrival, and write addresses respond much
more to the recent value than the earlier identifier. Source-swap read changes
are largely common across the two questions even during the first document.
Both checkpoints solve 5/16 diagnostic questions and one narrow paired swap;
all eight matched identifier-exchange pairs per checkpoint keep their answers
unchanged when value order is held fixed. This supports an early binding
weakness without proving information destruction or excluding routing limits.
Close the diagnostic. The completed [multiscale design review](local-multiscale-writer-review.md),
supported by [primary sources](local-multiscale-writer-sources.md), selects a
64-neuron writer bank: 32 update at 0.5, 16 at 0.125 and 16 at 0.03125.
Its local eligibility includes the corresponding input susceptibility and
self-path carry. Slower retention can still mean weaker absolute signals;
all maps retain generic access and the query cell stays unchanged. Validate
exact scalar-predecessor parity, fixed-vector state and the fresh complete cost
before one integrated comparison. The [completed multiscale return](../../experiments/r4_integrated/local_memory_multiscale_v1_result.md)
passes corrected saved audit and finishes in 374.542 seconds after admission at
862.514 projected seconds. Tagged, aggregate and the candidate frozen writer
each score 5/8 fit, 2/4 new-value, 2/4 delayed and zero interference; the scalar
reference exactly reproduces 5/8, 1/4, 1/4 and zero. One fresh case and its delayed
mate improve, but neither assignment-swap pair is jointly correct. The frozen
writer matches every tagged correctness decision, so the changed dynamics show
a modest prior benefit without learned-writing qualification. Preserve this
result and close the fixed fit. The [binding-architecture review](local-token-event-binding-review.md)
selects one [token-to-neuron event interface](local-token-event-interface.md):
generic byte runs capped at 16 bytes and a fixed 32-neuron recurrent encoder
shared by documents and queries. This shortens writer sequences while preserving
source bytes in the segmentation contract; neuronal codes remain lossy priors.
The [preprocessing qualification](../../experiments/r4_integrated/local_byte_event_preflight_v1_result.md)
now passes with independent audit in 1.089 seconds. Every source byte and span
is preserved; TRAIN document events project to 22,528 versus 51,200 byte/EOS
steps, and query events halve. No neuronal model ran, so these counts establish
neither a simulation speedup nor binding. The
[completed event-encoder comparison](../../experiments/r4_integrated/local_memory_event_v1_result.md)
passes fresh cost admission at 773.893 projected seconds and completes with
independent audit in 314.136 seconds. Every updating event arm, including
writes-disabled, solves only 1/8 taught checks and none of the new-value, delayed
or interference checks. The byte-stream reference exactly reproduces 5/8, 2/4,
2/4 and zero. Total run time is lower than the preceding comparison, but this
fixed representation regresses useful behavior and is not adopted. Close this
fixed fit. The completed [sequence-memory review](local-sequence-memory-review.md)
selects an explicit neuronal byte sequence, learned shared query-to-start
addressing and forward sequential recall. A smooth generic copying prior and
causal prefix beliefs give current teacher signals a direct retrieval path;
codec credit remains a declared local filtering approximation. Retire the old
writer/tag machinery from this main path. Literal sensory storage is counted,
not claimed as semantic compression or learned byte identity. Implement and
validate this one integrated replacement, then require fresh five-arm cost
admission before its fixed fit. The [completed sequence return](../../experiments/r4_integrated/local_sequence_memory_v1_result.md)
passes its fresh 518.720-second projection and independent audit, completing
the run and audit in 213.521 seconds. Learned scores 4/8 fit, 2/4 new-value,
2/4 delayed and 1/4 interference; each candidate control scores zero. The byte
reference exactly reproduces 5/8, 2/4, 2/4 and zero. One new-value assignment
swap is correct in both directions, while the other selects the wrong values;
all four different-question fit pairs still share an answer. Learning in the
addressing path contributes useful behavior, but reliable question-to-record
binding and the strict completion criterion remain absent. Close this fixed fit
and use its saved prefix beliefs, outputs and work receipts for one bounded
follow-up review before selecting another integrated change. The completed
[query-interface review](local-lag-query-review.md) selects a generic 32-by-32
byte-lag query population with fixed total squared activity 32 and one learned
1024-by-1024 address projection. This tests direct feature access and exact
local projection credit; saved outputs do not prove that recurrent compression
destroyed identity information. Keep tape and codec unchanged, preserve both
the current sequence and strongest byte references, and require ordinary QA
and fresh six-arm admission before one fixed comparison. Retire the obsolete
query cell, its traces and unused upstream feedback in this candidate. The
[completed lag-query comparison](../../experiments/r4_integrated/local_lag_query_v1_result.md)
passes its independent audit in 271.806 seconds including the run. Learned solves
7/8 fit, 4/4 new-value, 4/4 delayed and 2/4 interference checks; all candidate
controls score zero, and both historical references reproduce exactly. Both
novel assignment swaps now work in both directions. Three different-question
fit pairs also produce their respective correct answers; the fourth retains one
wrong answer. This is a useful query-binding and fresh-value gain, while the
strict completion criterion still fails on three cases. Close the fixed fit
without extra epochs or a gain sweep. Review the saved incorrect start ranking
and multi-record byte competition before selecting one next integrated change.
The completed [sharing review](local-shared-temporal-query-review.md) selects
one 32-by-32 temporal map reused across all 32 sensory-code channels. It learns
generic lag alignment with held local synaptic products, retaining tape, codec,
soft competition, rate and exposure. This supplies a shared channel-correspondence
prior; it does not establish the cause of the dense map's failures. Preserve
the immediate dense-map learner as the exact reference, complete ordinary QA,
and require fresh five-arm admission. Success requires absolute 20/20, novel
superiority to the three matched controls and no regression plus overall gain
over the dense reference, which already solves all four novel cases. This new
criterion does not change the closed result. No hard-address or gain sweep is
queued. Shared-address operation and storage counts shrink; actual whole-workload
time and memory must still be measured. The
[completed shared-map comparison](../../experiments/r4_integrated/local_shared_temporal_query_v1_result.md)
passes its independent audit in 119.745 seconds including the run. Learned now
solves 8/8 fit, 4/4 new-value, 4/4 delayed and 2/4 interference checks; all three
controls score zero and the dense reference exactly reproduces 7/4/4/2.
All four different-question fit pairs are correct, and the last 128 acquisition
first attempts are all exact. Strict qualification still fails on two interference
cases. Close this fixed fit; falling loss and perfect late first attempts do not
authorize more epochs. The saved correct individual starts still lose to pooled
bytes from several competing records. A proposed frozen single-address policy
comparison was reviewed but remains unselected and unrun. The user's renewed
no-backpropagation requirement instead triggered an
[actual-equation review](local-learning-backprop-boundary-review.md): the existing
teacher manually returns address error through the fixed key map. Withdraw it
from further training, including reference reruns. Its completed measurements
remain historical evidence, not a backpropagation-free capability claim.
The [local-learning return review](local-neuronal-learning-return-review.md)
now owns the next change: fixed query/key joint activity before a single learned
address-output layer, with direct local output error and no reverse key-error
path. Both free action and teaching must use that actual forward circuit.
Rejoin real executed first attempts and local feedback on the same learner
instance; the previous acquisition attempts were byte predictions, with action
execution measured only at the endpoint. Fresh acquisition and complete cost
are required. No frozen-checkpoint shortcut substitutes for this repair.
The [joint-address online cost](../../experiments/r4_integrated/local_joint_address_online_v1_result.md)
and independent audit pass in 21.506 seconds, but the complete projection of
1,056.720 seconds exceeds the 900-second ceiling. No full acquisition ran.
All sampled actions took the pending fallback, so successful subprocesses are
not successful model actions and accepted motor execution remains unsampled.
The [execution-speed review](local-action-execution-speed-review.md) selects a
qualified direct CPython launch using its established Windows virtual-environment
preservation mechanism, plus one explicit accepted-action cost probe. Keep the
new local neuronal rule, same-instance action/feedback order and fixed exposure;
require authored environment/output parity and fresh complete cost before fitting.
The [v2 cost](../../experiments/r4_integrated/local_joint_address_online_v2_result.md)
then stopped on a duplicate exclusive request-file publication, after sixteen
lessons and before worker/final evaluation. Its partial result admits no fit or
speed claim. Select the [v3 export repair](../../experiments/r4_integrated/local_joint_address_online_v3.md):
publish the complete request once, verify the composition with a non-neural
fixture, then require fresh cost with the same learner, executor and exposure.
The [complete v3 cost](../../experiments/r4_integrated/local_joint_address_online_v3_result.md)
and saved audit pass in 22.289 seconds, but project 996.224 seconds against 900.
No full acquisition ran. Review saved action cost and executor source to select
one bounded improvement; retain accepted-path coverage and the fixed workload.
First execute the [two-child authored executor profile](local-action-executor-profile-v1.md)
under unchanged transport, with no learner/motor calls or acquisition.
That profile places 94-95% inside subprocess. One [authored child import profile](local-action-import-profile-v1.md)
now localizes import subtrees before choosing a regular-source bootstrap change.
It found a 17-ms `pkgutil` subtree. Select the
[v4 source bootstrap](../../experiments/r4_integrated/local_joint_address_online_v4.md),
retaining the existing qualified main-module lifecycle and requiring authored
semantic QA plus fresh complete cost before learning. No neuronal rule changes.
The [complete v4 cost](../../experiments/r4_integrated/local_joint_address_online_v4_result.md)
and audit pass in 27.597 seconds but reject the 1,886.974-second projection.
The slowdown spans unchanged model work and action execution; no contemporaneous
host evidence establishes its cause. No full acquisition ran. Preserve this
failure and review concrete executor/simulation optimizations before selecting
one bounded next change; no identical timing retry or budget relaxation.
The selected [runtime return](local-online-runtime-return-v1.md) pairs an explicitly
standard-library action runtime in fresh processes with current-call held codec
responses. Qualify both boundaries and actual work/storage accounting, then
return once to the same integrated workload with fresh complete cost admission.
The [complete v5 cost](../../experiments/r4_integrated/local_joint_address_online_v5_result.md)
and audit pass in 21.232 seconds, but reject the 956.052-second final projection;
memory projects 702.879 MiB. No full acquisition ran. The supplied accepted-action
probe guards all four action maxima and spends 90.95% of its complete action time
inside the executor. Select one [current-runtime import diagnostic](local-stdlib-action-import-profile-v1.md)
to measure the remaining runpy subtree under the qualified no-site runtime.
That diagnostic found a 17.281-ms runpy subtree. Select the
[temporary-module bootstrap](local-temporary-module-bootstrap-v1.md), preserving
the current strict-output and main-module lifecycle with public APIs. Authored
semantic qualification and a fixed timing comparison must earn integration;
no neuronal rule change, identical cost retry or admission relaxation follows.
Both now pass: eight semantic pairs and a twelve-child authored comparison with
aggregate new/old execution ratio .7172. The prepared
[v6 return](../../experiments/r4_integrated/local_joint_address_online_v6.md)
keeps the learning workload unchanged and requires fresh complete cost plus
independent audit before any full acquisition.
Run/audit peaks of 665.539/432.832 MiB fit the revised admission estimates.
The shorter aggregate duration includes a changed reference roster and must not
be presented as an isolated circuit speedup.
Streamed scientific receipts cost 8.693 seconds and remove retained lesson-list
growth. Run/audit peaks are 907.887/988.676 MiB, below the final 1,617.741-MiB
admission estimate. The run-specific estimate still understates its peak by
75.406 MiB; its residual parent growth is not localized by saved receipts.
Do not infer convergence or a formal allocator bound.
No encoder learning
or output-only diagnostic is added. Segmentation must not
supply identifier/value roles or answer factors. Do not extend epochs or sweep
leaks, broadcasts or seeds. No special identifier
detector, supplied address, answer factors or protected gate follows.
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
