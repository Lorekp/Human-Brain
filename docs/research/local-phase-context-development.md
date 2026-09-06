# Certified phase/context reference: implementation contract

2026-09-06. Recorded before implementation, following
[the learnability review](local-decoder-learnability-review.md). The current
random and symmetric decoders fail exact composition; a memoryless bypass also
has contradictory inputs for repeated-symbol/EOS decisions. This reference
isolates output acquisition with an explicitly supplied temporal representation.

Implement one fixed CPU model: 96 binary context neurons, three clock neurons,
288 generic conjunction neurons (every phase and every context coordinate), and
257 byte/EOS output neurons. A fixed neuronal chain starts at BOS and advances
per decision. Features are `[phase, flatten(phase x context)]`, with no alphabet,
answer slot, target-derived selection or proof-witness weights in the model.
All output synapses start at zero int32 ticks; one tick represents 0.01.
An incorrect output winner causes the target neuron's synapses to add their
binary presynaptic activity and the winner's synapses to subtract it. Exact
integer scores determine the lowest-index tie winner. No update occurs on a
correct prediction or when learning is disabled.

Teaching always offers its three independent clock phases. Read-only generation
stops on learned EOS or the supplied byte cap (at most two), never a clock-forced
EOS. Contexts may be binary CPU float32/64; this reference is not a general
continuous-input or variable-length decoder. Supplied current steps are validated
against their current integer scores before teaching; delayed changed-score
steps are rejected rather than violating the immediate perceptron assumptions.

All learned and fixed synaptic arrays, clock initial state and thresholds are
serialized with exact configuration metadata. Externally modified fixed
dynamics, nonbinary/invalid inputs, exhausted clock use and int32 update overflow
must fail before parameter mutation. Caller-owned step/state tensors remain
independent; no hidden module activity, cache, optimizer or gradient exists.

Accounting distinguishes 644 neurons including the 96 supplied input neurons,
74,787 learned output synapses, and 585 allocated fixed synapses (nine clock and
576 conjunction connections, 578 nonzero). There are 288 fixed thresholds and
three clock-initial values. Serialized buffers occupy 302,652 bytes: 299,148
output-tick bytes plus 36 clock-weight, 12 initial-clock, 2,304 conjunction-weight
and 1,152 threshold bytes. Caller-owned phase state is 12 bytes; one saved step
contains 1,164 feature bytes, 2,056 int64 score bytes and the next 12-byte phase,
totalling 3,232 bytes. Supplied float32 context adds 384 bytes. `accounting()` and
state/step `tensor_bytes` expose these payloads. Python/container and temporary
allocation costs remain part of measured process memory; this is not peak RSS.

The 54-mistake certificate assumes zero initialization, exactly two active
context units, fixed three-phase inputs, immediate sequential mistake updates,
complete repeated TRAIN cycles and exact integer arithmetic. The reference-zero
and learning-disabled controls are outside that certificate. It guarantees TRAIN
learnability, not withheld combinations or reading. Representation and clock are
gifted neuronal priors. No theorem witness enters initialization or teaching.

Budget: one implementation and integration-owned focused tests. No full fit,
timing or new configuration is run by this implementation agent. The separately
reviewed bracket recipe owns the fixed 768-lesson endpoint, six trajectories,
controls, intermediate observations and 120-second/2,048-MiB ceiling.

Implementation is complete and Ruff/whitespace checks pass. Independent source
review confirms exact clock/conjunction dynamics, integer mistake updates,
learned EOS and certificate assumptions. Constructor buffers and generation
tokens explicitly select CPU without changing ambient Torch settings or RNG.
Integration-owned numerical/interface QA passes all 79 tests in the existing
decoder suite; the receipt is `.pytest-tmp/local-context-learnability-v1-qa.xml`.
No full learning experiment or timing was executed by this implementation agent.
