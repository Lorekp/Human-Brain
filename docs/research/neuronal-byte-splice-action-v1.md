# Neuronal byte-splice action: bounded source and learning design

Status: selected design review, 2026-09-06. No implementation or acquisition
protocol is admitted yet. The [roadmap](roadmap-reading-to-coding.md) owns
priorities; the completed [v6 result](../../experiments/r4_integrated/local_joint_address_online_v6_result.md)
remains closed at 18/20, with useful no-backpropagation learning and two
interference failures. No recall-policy sweep or more epochs is selected.

## Observed gap and one direction

V6's neuronal system learns the replacement value, but its host supplies the
request, escaping and occurrence-selection motor. Useful byte recall now warrants
removing that supplied action choice. Develop one typed STOP/SPLICE interface:
neurons choose action type, file/start boundary, end boundary and raw replacement
bytes/EOS. The host authenticates the workspace and validates/applies precisely
those selected bytes. It must not choose a matching occurrence, escape content,
repair an invalid proposal or fall back to a correct edit.

Use half-open raw byte offsets and every file boundary, including EOF. A joint
file/start output and sequential end output avoid quadratic span enumeration.
Restricting end to the selected file and end >= start is a declared structural
prior. Input encoding must not parse semantic roles or filter to plausible answer
spans. Raw path context must remain available independently of local file context;
the existing 32-lag query cannot silently become an unbounded repository encoder.

This typed interface supplies tool serialization and boundary indexing as explicit
priors. It does not establish learned textual patch grammar, arbitrary Python
synthesis, planning or language comprehension. The current adjacent-source byte
codec supplies literal payloads only within its declared limits. Escaping/code
syntax required by an example must be emitted by the learner or fail visibly.

## Source/math review deliverables

Fix feature dimensions, path/request/file bounds, start/end conditioning,
operation competition, output caps, held-state lifetimes and numerical update
order before implementation. Check representational collisions and whether the
features distinguish different correct locations after fresh layout, path and key
swaps. Limitations stay exposed; a teacher-supplied free input or hidden candidate
filter must not resolve them.

Each learned score neuron directly consumes its actual presynaptic features.
Local teaching uses its own output error times those held features. No backward
error through fixed or learned upstream maps is allowed. The same model object
and synapses must propose the actual first action, receive execution feedback,
then learn from declared correction. Teacher-forced later fields remain separate
from free decisions. Grades remain evidence unless numerical use is explicitly
defined; supervised correction is not autonomous recovery.

Assess a declared transferred v6 reader prior, identical in every arm, with fresh
action synapses. Do not load it during this review. Specify matched learned,
action-frozen, all-slow-frozen and retained-memory-disabled controls, fresh
layout/key/value transfers, explicit STOP, invalid edits and execution failures.
Preserve delayed facts/interference and reading retention. Fix exposure, success
criteria and cost probes prospectively in the eventual experiment.

Count neuronal populations, persistent and transient tensors, feature creation,
direct local teaching, serialization and every real execution. Derive O(BD)
resource accounting and exact efficient vectorization before fresh complete cost
admission under 900 seconds and 2,048 MiB. V6's cost does not price new action
populations. Authored arithmetic/transport QA must precede acquisition and
distinguish hand-set mechanics from learned skill.

The orchestrator owns this single interface. Parallel agents review concrete
topology, locality/evaluation and resource feasibility; reports inform one
implementation contract. No model/forward/checkpoint/action calls or alternative
architecture experiments belong to this bounded review.
