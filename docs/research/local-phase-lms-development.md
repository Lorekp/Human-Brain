# Continuous phase/context LMS: implementation contract

2026-09-06. Recorded before implementation. Follow the selected
[composition review](local-composition-return-review.md): the integer perceptron
fits TRAIN but ceases updates before its output potentials identify the missing
compositions. Do not alter that closed reference or add further epochs to it.

Implement one new fixed `PhaseContextLMS()` CPU model. Its 96 finite signed
context rates feed all 288 generic phase/context multiplication neurons. An
explicit three-neuron chain supplies phase, and all 257 output neurons receive
`[phase, phase * context]` (291 features). Clock and gating priors are supplied,
not learned. No alphabet mapping, target-derived factor selection or proof-only
separator appears in runtime. Accept CPU float32/64 contexts and cast to float64;
all learned synapses start at float64 zero.

At each immediate teacher offer, use saved pre-update potentials `z` and the
all-neuron teaching residual `r = onehot(target) - z`. Every output synapse
updates by `0.01 * r_post * feature_pre`, including when its greedy winner was
already correct. Report loss as **half the sum of squared potential residuals**
over all 257 neurons, separately from previous cross-entropy and mistake losses.
No softmax, optimizer, BPTT, clipping, decay or hidden update is involved.

State and steps belong to the caller. Validate input, fixed dynamics, supplied
step features/phase, current potential equality, finite residual/loss and every
proposed synapse before publishing changes. Reject changed-score delayed steps.
Disabled teaching still computes and reports its offered residual but changes
nothing. Generation resets its neuronal clock, uses learned potential winners,
and stops only on learned EOS or the requested cap of at most two bytes. Teacher
traversal remains independent of an earlier free-generation stop.

Serialize learned/fixed synaptic arrays, initial clock and exact configuration.
Accounting covers 644 neurons (96 supplied context, three clock, 288 gating,
257 output), 74,787 learned output synapses and 585 allocated fixed synapses
(nine clock and 576 gating connections, 578 nonzero). Serialized tensors total
602,952 bytes: 598,296 float64 output-weight bytes, 4,608 gating-weight bytes,
36 clock-transition bytes and 12 initial-clock bytes. Independent clock state
occupies 12 bytes; a complete saved step occupies 4,396 bytes including that
next-clock state, 2,328 feature bytes and 2,056 potential bytes. The returned
residual adds 2,056 bytes; supplied float64 context adds 768 bytes. These are
payload counts, not process RSS; Python, snapshots and proposal scratch remain
covered by process-memory measurement. No module
activity cache, ambient RNG/settings mutation, fake embedding interface or
upstream context teaching signal is added. Actual-memory integration requires
its own explicit feedback and narrowed output-length contract.

The prospective 160-cycle LMS bound assumes exact real arithmetic, binary
two-factor inputs, fixed full-roster cycles and zero initialization. Float64
execution and arbitrary continuous document contexts do not inherit that
guarantee. Numeric output/margin qualification and saved-weight verification
belong to the separately frozen experiment; no witness or probe feedback enters
learning.

Budget: one implementation and integration-owned focused QA in the existing
test file. No fit, timing, parameter sweep or full diagnostic is run by this
implementation agent. The reviewed experiment owns four arms, one 1,920-lesson
endpoint per arm, and its 120-second/2,048-MiB total ceiling.

Implementation is complete. Root source review confirms that residual allocation
preserves caller potentials, fixed clock/gating matches the declared features,
all-output plasticity follows the local residual rule and finite validation
precedes weight publication. Ruff and whitespace checks pass. Integration-owned
QA passes 85 tests, with the six new cases also passing after strengthening the
independent scalar oracle. Receipts and disposable exposure counts are recorded
in the [experiment recipe](../../experiments/r4_integrated/local_context_lms_v1.md).
No model fit or timing was executed by this implementation agent.
