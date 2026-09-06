# Reusable project goal

Updated 2026-09-06. Paste the following into `/goal` when authorizing project
execution. Saving this prompt does not start a goal or authorize implementation
during a documentation-only task. The [roadmap](research/roadmap-reading-to-coding.md)
owns priorities, architecture policy, and evaluation order.

```text
Build an intelligence network for real-world use, including coding, that learns
continually, retains and uses knowledge, discovers reusable abstractions, and
improves through action and feedback. Keep simulated-neuron learning as the core
research commitment for now. You may modify the neurons, replace GLIF, change
learning rules, and grow or redesign the network.

The main learner must learn online through local cell and synapse rules using
its own state, pre/post activity, eligibility traces and declared teaching,
reward or error signals. Do not conceal reverse-mode gradients or externally
computed per-synapse updates inside those signals. No backpropagation, BPTT,
Adam/global fitting or manually coded reverse-Jacobian transport is authorized.
Learning and acting must use the same persistent neuronal system.
Repeated experience and explicit replay are allowed with
their exposure and costs reported. Demonstrate language and context learning in
the neuronal system; locality alone is not evidence of understanding.
Vectorize faithful implementations of these local rules when useful for speed.

Read docs/research/roadmap-reading-to-coding.md now and after each context
compaction. Follow its current milestone and linked plan rather than obsolete
queues. Develop a broader persistent reading-to-memory-to-executable-action
workload now; do not require perfection on the existing toy cohort. Prioritize
unseen instruction forms, structural combinations, longer memories and reduced
teacher dependence. Report supervised acquisition, fresh transfer, retention
and teacher-free recovery separately, with matched learning-disabled and simple
retrieval/editing references. Keep old tasks as regressions and move experimental
chronology out of the main roadmap. Use authentic public
exam reading comprehension, with official questions and marking guidance, as
evaluation pressure. LLM-authored questions are not a substitute for that evidence.

The system need not be cell-only. Tokenization, external/hybrid knowledge stores,
published research, and pretrained LLM components, including base/raw models as
possible knowledge resources, may be investigated. Make the neuronal core's
learning and causal contribution measurable against matched controls, including
the external component alone. Disclose every component and its role; avoid
turning the project into an ordinary LLM wrapper or a collection of task-specific
solvers. Preserve the original contracts of frozen experiments.
External knowledge inputs remain allowed, but outsourcing understanding or
action choices and merely gating/copying the result does not establish the
core's capability. Keep any assisted result distinct from neuron-only evidence.

Treat efficiency as integral to the system. Optimize measured end-to-end
bottlenecks so useful learning can scale. The suggested 100-billion-neuron scale
is exploratory, not a required count: justify expansion with capability and
resource curves, and distinguish addressable from active and useful neurons.

Progress from simple programming exercises and small real repositories to
protected transfer and larger SWE benchmarks in the roadmap's gate order. The
long-run comparison targets are Gemma-4-12B (quantized allowed) on a fair SWE
benchmark, then Opus 4.8 on DeepSWE. Verify exact model and benchmark identities
and availability before preregistering the comparison; do not silently substitute
a different target. Match and report access, tools, attempts, data, and resource
budgets. Keep protected evaluation separate from development and prevent
benchmark overfitting.

Orchestrate independent subagents liberally where parallel work can advance the
shared milestone. Use high reasoning effort whenever using GPT-6 Astra as a
subagent. Keep the high-level dependency picture, review returned work, and
dispatch useful follow-ups when they become unblocked. Avoid duplicate efforts
and competing architecture queues; continue independent work while others run.

Before each mechanism experiment, state the integrated failure it addresses,
the hypothesis, success criterion, and budget. Return to integration after its
focused test. If repeated failures or an exhausted budget leave the approach
unproductive, dispatch a bounded research/reasoning review and compare other
approaches. Preserve honest negative evidence and replace ineffective designs.

Stay motivated and persistent toward the ambition. Treat setbacks as information
for the next decision; judge progress by useful learning, transfer, retention,
recovery, and measured efficiency rather than activity or optimistic claims.
```
