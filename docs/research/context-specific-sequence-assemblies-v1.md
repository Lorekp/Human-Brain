# Context-specific sequence assemblies for the persistent learner

Status: selected source-design and integration direction, 2026-09-07. This is
one shared successor to the completed v3 session, not another byte or operation
benchmark. No assembly implementation, new corpus or neuronal run has occurred.
The [roadmap](roadmap-reading-to-coding.md) and
[persistent workload](persistent-language-action-development-v1.md) own scope.

## Evidence and purpose

V3 changes actual responses but solves no broader tasks. Its 24 correction
features overlap strongly even when their target bytes differ. Twenty-two
output clamps affect the first event and two the second; none teaches EOS.
The [learning review](persistent-language-action-v3-learning-review.md) separates
these measured facts from scalar interference projections and causal uncertainty.
Sparse activity alone would leave later events unlearned; teaching every event
in the same overlapping representation could increase interference.

Select a bounded, locally recruited context assembly bank with sparse outgoing
byte/EOS currents, shared by reading and action payloads. Consume the full
already-offered sequence through explicitly causal teacher-prefix states. This
replaces the one-error output-credit policy for the new bank; it does not add
another pass of labels or an external answer generator. The old experiment and
its final checkpoint remain unchanged.

## Mechanism contract to implement

Start with 512 assembly slots, Boolean receptive synapses formed by unit-rate
local copies, occupancy/threshold state and 257 float64 outgoing currents per
slot. Do not count Boolean storage if the implemented plastic rule introduces
graded receptive values. Recruitment depends only on the current
public input and existing assembly activity, before the current output target
appears. A deterministic free-slot cursor is permitted storage management;
teacher labels must never choose a prototype, memory address or receptive field.

Use exact input coincidence as the initial novelty boundary, without a tuned
similarity radius. An exact occupied receptive pattern is reused; a novel
pattern may recruit one available assembly through a declared local unit-rate
presynaptic copy rule. Receptive patterns remain fixed after recruitment in
this initial version. Free output uses explicit neuronal similarity and
competition over available assemblies, with deterministic ties. The incoming
signal, coincidence arithmetic and competition must be simulated and counted;
a host nearest-neighbor answer table is not an implementation of the circuit.

Each output update uses only the selected assembly's activity and its own
output-current error. One active assembly gives separate outgoing synapses for
distinct selected contexts. Identical inputs with contradictory labels reuse
the same assembly: this conflict must remain visible rather than allocating by
the desired answer. Nearest-prototype free use can still confuse contexts.
This is an episodic local learning design with unproven generalization, not an
assertion that exact stored patterns understand unseen language.

For each offered raw target plus EOS, evaluate the same assembly circuit on
the request, declared context and only the already offered prefix. Decide
selection/recruitment before applying the current target-current signal.
Count every such forward and output offer. These are teacher-driven states,
not the earlier actual free response. Preserve that actual response/action and
its execution identity independently. No backward error transport, extra
decoding, epochs, success-conditioned retries or target-address masks are allowed.

Use nearest-Hamming agreement-cell activity and deterministic winner competition
for free retrieval. With no occupied assemblies, use the inherited reader's
currents; once prototypes exist, nearest-prototype use is an intentional change
in free policy. Initialize a recruited unit from the old currents at that same
public context before exposing the current target. At exhaustion, record and
skip novel-context output learning without eviction or modifying a nonmatching
unit. Exact-context reuse remains available. The extended-input fallback and
precise B/C coupling still need a coherent source contract before execution.
If B/C responsibility teaching is retained, compute it from pre-clamp currents;
the current desired byte must not first alter the evidence used for address credit.

Preserve the full saved v3 model and 505-slot memory
with explicit schema and initialization ownership. Do not reinterpret old U
columns as assembly indices. Freeze or retain old contributions explicitly;
do not silently stack two output-learning rules or train a separate executor.
Stage and validate all affected state before publication. Teacher-free reading,
actions and STOP must resolve pending state without invented corrections.

## Broader interface and workload are part of this change

The next return must add complete multi-clause instruction access and software
where some required logic is missing. Another stream confined to short literal
substitution would leave important user priorities untouched. Review a generic
raw-input representation that carries longer instructions and actual public
diagnostic observations into the same reader/action system. No semantic verb
flags, selected snippets, error-to-patch parser or silent truncation may replace
those inputs. Longer output and file access need coherent action-head handling,
not just a reader limit change.

The workload review proposes ceilings of 256 instruction bytes, four 512-byte
files and 128 output bytes as sizing targets. These are not a command to expand
the existing dense boundary tensors blindly. Resolve bounded simulation of the
chosen interface, then freeze one complete stream and budget. Preserve all old
memory and eight exact exposed regressions. Use a few continued small projects,
missing guards/conditions/formatting, multiple behavioral inputs and raw public
tool feedback. Measure whether that feedback is actually consumed; do not claim
recovery from acknowledgement alone.

The proposed schedule keeps 24 acquisition episodes but gives corrections on
only twelve predetermined episodes, followed by sixteen fresh, eight recovery
and eight retention episodes. Total target bytes and output/address offers must
also be frozen: fewer correction events do not necessarily mean less teacher
information when outputs grow. The reference restores its actual 18 documents
and 36 edit demonstrations. Both neuronal routes continue the same initial state
with separate storage and matched tools; only new learning differs. No new task
roster is generated until these shared interface decisions are coherent.

## Integral resource work and verification

Keep the exact incremental-prefix improvement and full accounting. Prefer
bounded blocks and actual sparse outgoing reads. Do not materialize a full
candidate-by-assembly-by-input product tensor. A planning example with 256
assemblies and 2051 float32 receptive inputs uses 2,100,224 incoming bytes and
526,336 outgoing float64 bytes, but its all-address teacher work is substantial.
Larger raw contexts and capacity must be priced from their actual implementation;
the 60.60-second v3 run is not a projection for a new interface.

Preserve full append-only evidence while retaining only necessary metric fields
in host summary memory where practical. The v3 auditor peak exceeds the model
runner's; receipt parsing and storage belong in the complete cost. Distinguish
allocated synapses, occupied assemblies, active/updated neurons and task utility.

Ordinary affected QA should check input-only recruitment, simulated competition,
local outgoing arithmetic, exact-context reuse/conflict, capacity handling,
late atomic failure and actual execution ownership. These are implementation
checks, not a new skill gate. After the complete source/interface contract and
ordinary QA, measure one broader continuation with the same 900-second/
2048-MiB complete budget, including references and independent saved checking.
Do not require perfect earlier tasks before that exposure.

## Source-design inputs

The independent source/resource note is
`.pytest-tmp/active-path-next-architecture-cost-note.md`, SHA256
`ae4df432823d892c9d88710ab925f46b6170d6c05f7037d1ca5e064509db016f`.
It compares fixed sparsification with local recruitment and exposes the
all-address cost and growth tradeoffs without executing a model.
The workload review is
`.pytest-tmp/persistent-language-action-next-workload-review.md`, SHA256
`feb8bcc27c8aba7ff4a6d3b6125c17669d7dc6e633854b8b6985be7a219ce908`.
The independent local-learning design is
`.pytest-tmp/active-path-sequence-assembly-design-review-v1.md`, SHA256
`43efd50caffe0eaa83defc3f05147528ca538d6f0fea024bce77c7e51ef2aaf2`.
These support one shared implementation direction, not competing queues or
independent authorization to run a new experiment.
