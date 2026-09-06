# Neuronal byte-splice action: bounded source and learning design

Status: source/math review accepted for implementation and authored QA, 2026-09-06.
No acquisition protocol is admitted yet. The [roadmap](roadmap-reading-to-coding.md) owns
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

The orchestrator owns this single interface. The completed source/math reviews
made no model/forward/checkpoint/action calls. The selected
[concrete topology](neuronal-byte-splice-topology-v1.md) and
[resource ledger](neuronal-byte-splice-resource-v1.md) now fix implementation:
6177 start, 2049 end and 4244 operation features, 16716 fresh float32 parameters,
four files of at most 128 bytes, 31-byte request/path bounds and 16-byte payload.
All boundaries remain eligible. The same raw request reaches the reader; no
host key extraction or guaranteed transfer of its old query distribution.

Operation is selected last, including after capped payload generation. STOP
teaches only that output on the actual proposal; corrective SPLICE teaches its
explicit corrected start/end/payload prefix after execution. Every action
proposal must validate before publication. Precompute and validate all action
parameter proposals, use the existing reader's atomic teacher publication, then
copy the already validated action proposals without subsequent fallible checks.
This keeps one learner and avoids a separately trained clone. Operation's bias
augments its normalized feature energy bound from 32 to 33; eta remains .01.
Read-only evaluation consumes its pending attempt through an explicit completed
execution acknowledgement without teaching. It must reuse the same identity and
receipt checks, clear only the pending ticket, and make no neuronal forward,
teacher or parameter update. Final evaluation must not require a fake correction.

Reuse OpaqueWorkspaceFile with explicit portable path validation and canonical
raw-path ordering. A small immutable SpliceWorkspace supplies its digest and
every (path, offset) boundary. SpliceProposal retains actual provisional fields
even for STOP; apply_splice validates the digest and applies only the selected
SPLICE bytes. It takes no teacher, expected output or semantic target argument.
Keep qualified executor source unchanged, outside neuronal selection.

Implement this source plus about eight to twelve focused existing-file authored
QA nodes: independent scalar arithmetic, raw-boundary/overflow/alias properties,
free versus corrected prefixes, modes and atomicity, STOP/empty/capped payload,
exact raw application and one real same-instance action-before-teacher trace.
Review actual authored exposure and a clean checkout doctor before execution;
never select old backpropagating nodes or a broad historical suite. These are
mechanics checks, not fitted capability. A new experiment must separately freeze
its roster and receipt size bound: seventeen payload decisions can exceed the
old two-byte receipt guard. No fit, timing probe or gate advancement is admitted.

## Reviewed implementation and prospective QA

The learner source is held at SHA256
`bb06e074d44431cf2d09d62f7e98119a2a207ed53e7f6d5796bb6c7b274770ab`;
raw transport is `9a704b4f9527598cbd784f2eaee9472e44a968eb21f97ceabf2ad2dfeb0de13a`.
Root and independent source reviews pass. File-invariant products are reused
before full feature normalization; teacher loss's additional log-softmax is
counted. Snapshot restoration preserves numerical state but cannot rewind the
live instance's action receipt counter. No source review ran a neuronal model.

Eleven appended QA nodes are held in the existing decoder test file, SHA256
`3d50d11b2f28c7c2b9b39cf66ce6b8ea5be682e6135210a0c5636a479fa6d8ba`.
All 138 previous top-level AST nodes are unchanged. Their prospective exposure
is ten combined models, two action-only circuits, nine tiny fact ingests,
eleven free proposals, seven successful SPLICE corrections (21 reader byte/EOS
offers), one rejected atomic correction with zero reader calls and one STOP
correction with zero reader calls. There are 28 attempted direct action labels
including three scalar fixtures; the late fault prevents one completion counter.
Free payload decisions are outcome-dependent, at most 187, and must not be
reported as that many actual decisions. Exactly one real Python child and zero
old motor calls are planned. No trained prior checkpoint is loaded in QA.

After freezing source and doctor, run the single
`test_splice_same_instance_execution_before_correction` node first, then the ten
remaining explicitly named nodes if it passes. The first fixture verifies the
actual resulting workspace once, including unchanged files after STOP/rejection;
that verification is host testing, not a neuronal RUN or substitute edit.
Other outcome mappings are explicitly simulated arithmetic/interface fixtures.
Archive exact tested source and retain any failure before repair. Ordinary QA
does not authorize acquisition or earn a learned-action capability claim.
