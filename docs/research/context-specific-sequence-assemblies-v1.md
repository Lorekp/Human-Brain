# Context-specific sequence assemblies for the persistent learner

Status: selected source-design and integration direction, 2026-09-07. This is
one shared successor to the completed v3 session, not another byte or operation
benchmark. Assembly, observation and integration source is implemented, and
[ordinary component/lifecycle QA passes](sequence-assembly-integration-qa-v1.md).
The broader corpus and continuation are frozen and released for one monitored
run; no new neuronal workload result is claimed at release.
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
unit. Exact-context reuse remains available. The source contract below freezes
the inherited B/C/U and codec while the assembly bank learns. There is no
all-address responsibility teacher sweep in this version.

Preserve the full saved v3 model and 505-slot memory
with explicit schema and initialization ownership. Do not reinterpret old U
columns as assembly indices. Freeze or retain old contributions explicitly;
do not silently stack two output-learning rules or train a separate executor.
Stage and validate all affected state before publication. Teacher-free reading,
actions and STOP must resolve pending state without invented corrections.

## Broader interface and workload are part of this change

### Selected raw-input and inherited-state contract

The new version accepts at most 256 request bytes, four files of at most 512
bytes each, paths of at most 31 UTF-8 bytes and 128 response/replacement bytes.
Each actual public stdout and stderr stream has a separate 512-byte ceiling;
overflow remains an explicitly incomplete capture. Eight signed little-endian
return-code bytes preserve negative codes and Windows DWORD values. Public
presence, execution acceptance, timeout and capture/file completeness flags are
inputs; correctness grades and teacher fields are excluded.

The observation frame has 3471 symbol positions, each represented by 18 Boolean
rails, and twelve flags. Separate fixed fields preserve every path, file,
request and captured diagnostic byte with EOS and distinguish absent padding.
The assembly adds BOS plus the full prefix in 129 symbol positions, the selected
64-row memory context and 129 position cells: 66,093 Boolean inputs in total.
This allocates 33,839,616 receptive bytes and 1,052,672 outgoing bytes, plus eight
occupancy bytes per reader. A private teacher stage duplicates this bank.

Memory addressing retains the old B/C matrices and all 505 occupied slots.
Every mechanical 31-byte request chunk receives EOS and its own original
normalized coincidence bank. The arithmetic mean of these banks, in source
order, drives B; C receives exact rolling activity for the most recent 32
prefix symbols. The complete prefix also enters the assembly. No chunk or
memory address is chosen from a teacher label. Chunk averaging loses some
cross-chunk order in address selection; the assembly's raw observation preserves
byte positions. Tool text enters assembly and action inputs, rather than the
old request-address matrix.

The inherited conditional U contributes fallback currents only where its
original geometry applies: request length at most 31 and output position at
most 16. Elsewhere the inherited codec supplies currents at the selected memory
address. There is no truncated request/prefix passed to U. New assemblies begin
from these same-context currents and then own their outgoing currents. This
changes the free policy and does not claim old-trajectory parity. B/C/U and the
legacy codec are frozen during new teaching, preserving context selection from
drift due to the current lesson's address credit. No second output rule is stacked.

Action heads use the same request and actual diagnostic fields. Their 1291
query symbol positions plus seven status-as-byte code rows give 1298 query
rows. Candidate contexts remain mechanical raw file boundaries. Start/end/
operation feature widths are 748947/42561/149252; fixed blocks of sixteen
candidates avoid a full candidate-by-feature bank. Teaching uses two passes:
all candidate currents/softmax first, then the same fixed features and their own
local output error products. Prior action weights embed by explicit old row and
column coordinates; new coordinates start at zero. Original offset/span `/128`
and payload `/16` scales retain their meanings, with larger attainable ranges.
Expanded normalization and reductions mean old numerical trajectory parity is
not claimed. The complete new workload must measure this larger simulation.

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

The reviewed v4 source now freezes that schedule: six interleaved reading/edit
pairs receive one correction each; six related pairs receive none. Three actual
project chains continue through changed requirements. Missing conditional,
collection, guard and formatting logic is checked through multiple public
driver inputs. Parent-only nominal solutions never replace a route's actual
files. The old eight v3 retention items remain exact exposed regressions.

The first pure manifest build has SHA256
`9d3399a1360ca9ddb4ba9361f06b70deb6063ddcf6461ed1aa414b0d54f4f191`
at `.pytest-tmp/persistent-language-action-v4-manifest.json` (84,120 bytes).
It ran the builder once and no model, checkpoint load, baseline prediction or
authored program. Build receipt SHA256 is
`2092c8ceb262bfee9ddcc1c6f0c8bc9e8beb2c5ed3d1af86bab5c648f7ebc34f`.
Twelve corrections offer 223 raw bytes plus twelve EOS events: 235 neuronal
reader offers and eighteen action labels per route. Fewer correction occasions
therefore do not mean fewer target symbols than v3's 172. All 48 new requests
have 117--168 bytes; the eight exact earlier regressions keep their original
requests. The whole 505-slot tape remains unchanged; context assemblies add
episodic capacity, without a claim that tape capacity or knowledge has grown.

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

## Release review

The six distinct ordinary mechanics/lifecycle QA nodes passed, followed by one
affected combined-node repeat after receipt and accounting changes. Independent
source reviews of initialization, the reference, actual project continuation,
the runner and saved auditor found no remaining launch blocker. No experiment
checkpoint was loaded by those checks. The frozen manifest above was built once.
The [source resource envelope](persistent-language-action-v4-resource-plan.md)
selects one 840-second run plus a 60-second saved-only audit under 2048 MiB.
This release measures the whole broader workload without an additional cost
probe or toy skill gate. The final source closure binds all released Python
files, including the saved auditor and retained-reader custody assertions.

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
