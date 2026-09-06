# Instantaneous symmetric hidden feedback: bounded development

2026-09-06. Contract recorded before implementation. The direct-context v1
diagnostic completed and passed its saved-evidence audit, but failed its positive
criterion: correct-context generation reached 4/12 trained and 0/4 held-out
pairs, while other arms scored zero. Source review found no sign, gain,
self-path eligibility or teacher-forcing implementation defect. Poor alignment
of the fixed random hidden broadcast remains one testable approximation limit.

Implement one explicit `ByteSequenceDecoder` constructor choice:
`hidden_feedback_mode="random"` (default) or `"symmetric"`. Symmetric mode
requires local learning. Serialize the mode in checkpoint configuration and
reject configuration mismatches; introduce no checkpoint compatibility path.
Keep every initialized tensor identical across modes, including the retained,
serialized and counted random hidden-feedback buffer when it is inactive.

For the saved current-byte error `delta = p - onehot(target)`, retain the exact
existing random expression or use `F.linear(delta, output.weight.T)`. Compute
this before **any** current-byte cell or readout parameter update. For immediate
teaching this is the exact instantaneous derivative of the linear readout loss
with respect to hidden activity. Delayed teaching uses saved error and current
readout weights; it does not reconstruct an old readout derivative.

The symmetric choice explicitly transports the forward readout weights along
a return signalling path from output neurons to hidden neurons. This is a
neuronal learning signal, with each synapse still applying its own local
eligibility; it is not an independent per-synapse external update. It adds weight
transport and makes no biological-fidelity claim. Recurrent self-path traces
still omit cross-neuron paths and intervening parameter-update derivatives;
the complete learning rule remains approximate. Do not apply the nonlinear
gain again: the existing eligibility already includes it.

Cell dynamics, local traces, update rate/sign, readout update, fixed embeddings,
random context feedback and teaching event order remain unchanged. No autograd,
BPTT, optimizer, new retained trace or `LocalMemoryLearner` integration is added.

Budget: one implementation and focused numerical/API review, with tests owned
by the integration agent. Require float32/64 pre-update signal arithmetic,
unchanged random behavior, matched initialized tensors at seed 735, mode
round-trip/rejection and fixed-buffer ownership. No learning experiment or
timing is authorized by this note. A separately frozen direct-context v2 recipe
owns the single matched comparison, exposure and runtime ceiling. A focused
pass earns return to the integrated workload, not a feedback-variant sweep.

Implementation is complete. Independent source review confirms the symmetric
signal precedes all current-byte proposals/updates, default random arithmetic
and context feedback are unchanged, and mode metadata rejects silent cross-mode
restore. Ruff and whitespace checks pass. Integration-owned numerical QA now
passes 73 existing-suite cases, including independent scalar updates, both
floating types, pre-update signal timing, read-only parity and mode custody.
Receipts and discarded QA exposure are recorded in the
[v2 recipe](../../experiments/r4_integrated/local_context_use_v2.md).
No learning experiment, timing or acquisition claim belongs to this
implementation handoff.
