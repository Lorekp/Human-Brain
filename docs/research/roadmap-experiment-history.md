# Roadmap experiment history and retired directions

Status: historical companion to [`roadmap-reading-to-coding.md`](roadmap-reading-to-coding.md).

This document owns the chronology, failure localization, and retired-mechanism
history that used to make the main roadmap grow with every experiment wave.
The roadmap should link here rather than reproduce this diary.

For the current evidence table and exact claim boundaries, see
[`../benchmark-status.md`](../benchmark-status.md). For the implementation-level
program map, see [`../program-map.md`](../program-map.md).

## 1. Reservoir / readout era — closed

The original reservoir-computing family was systematically tested rather than
quietly abandoned.

- Multi-timescale eligibility traces and follow-on state-feature families failed
  to create useful context separation at the required capacity.
- Race/broadcast improved output diversity without producing correct QA.
- U-STATE/SDR storage demonstrated that separation and storage could succeed
  while generalization to unseen question-context pairings remained zero.
- APICAL-AND and other fixed-connectivity representation changes did not close
  the missing mapping between question space and content space.

The durable conclusion is that **capacity and separation are not sufficient**.
A useful learner needs learned relational/long-range binding between different
content spaces. Retuning presentation schedules, codebook geometry,
replay-prioritization, EI fraction, or old ridge/readout constants is not
licensed.

Key references:

- [`r0_multitrace/`](../../experiments/r0_multitrace/)
- [`r0b_state_features/`](../../experiments/r0b_state_features/)
- [`race-pivot-gate-verdict.md`](race-pivot-gate-verdict.md)
- [`custom-architecture-directions.md`](custom-architecture-directions.md)

## 2. Path-2 reading / persistent identity

The reading lane moved from an invalid monolithic identity criterion toward a
factorized persistent-successor criterion.

Important historical boundaries:

- calibration-v3 exposed candidate/matching versus active-path distinctions;
- a long public-TRAIN replay authenticated roughly 976k exact prefix steps before
  a later allocation criterion failed;
- the failure was reclassified when a repeated source sharing one output column
  with a novel successor showed that native identity is factorized by output
  column rather than `11 * novel_keys`;
- a target-independent fixture now pins stable complete tuples, stable exact-
  source owners per output column, and terminal storage equal to actual
  allocation sum.

The completed dense TRAIN build reached all 49 books / 2,117,412 tokens, but it
is not itself a capability verdict. The currently licensed adjudication is the
fresh source-bound factorized replay from empty native and scorer state:

- [`run_p2_factorized_successor_train49_replay_v1.py`](../../experiments/r3_pilot/run_p2_factorized_successor_train49_replay_v1.py)

Protected held-out/C1 material remains unopened.

## 3. Coding: whole-sequence continuation → action learning

The early coding bridge established useful plumbing but not general synthesis.

- PAC probes showed execute → observe → route → build loops on narrow TRAIN
  templates.
- The frozen fresh four-task coding cohort produced valid source but collapsed
  onto two learned whole programs: `STOP_FRESH_CODING_CHALLENGE`.
- Static prompt-to-production decoders could replay teachers after representation
  repair but still failed unseen composition.
- Blind edit search and the first persistent sensorimotor actor were closed after
  honest STOPs.
- Reward-modulated action learning localized magnitude aliasing; exact signed-
  magnitude representation then produced 10/10 atomic actions.

These experiments retired whole-program retrieval and one-step greedy editing as
credible general mechanisms.

Representative artifacts:

- [`fresh_coding_challenge.json`](../../experiments/r3_pilot/fresh_coding_challenge.json)
- [`direct_production_cell_composition_probe.json`](../../experiments/r3_pilot/direct_production_cell_composition_probe.json)
- [`direct_production_cell_role_gated_probe.json`](../../experiments/r3_pilot/direct_production_cell_role_gated_probe.json)

## 4. Learned successor planning → typed AST planning

The next architecture learned primitive transition consequences and used them to
plan.

- Visible scalar H=2 successor planning passed 24/24 mixed-order tasks.
- The fresh k=3 pair transferred the correct order but exposed unsupported
  hypothetical states, motivating an explicit physical `UNKNOWN` state.
- Typed AST site planning passed visible one-edit tasks across multiple depths.
- Genuine H=2 typed-AST planning recovered noncommutative order and real
  child-to-root replanning.
- Several beam/anytime variants localized physical tie and STOP-frontier issues;
  the final STOP-bounded anytime variant preserved the exact visible result with
  less work.

Representative artifacts:

- [`public_typed_horizon2_successor_visible_probe.json`](../../experiments/r3_pilot/public_typed_horizon2_successor_visible_probe.json)
- [`public_typed_horizon2_successor_fresh_k3_probe.json`](../../experiments/r3_pilot/public_typed_horizon2_successor_fresh_k3_probe.json)
- [`public_typed_ast_horizon2_int_visible_probe.json`](../../experiments/r3_pilot/public_typed_ast_horizon2_int_visible_probe.json)

## 5. Factorized typed edits → broad bridge STOP

Factorizing site/schema/operator/operand/literal improved controlled transfer.
H=1 factor transfer passed renamed bindings, changed prompt surface, signed
literals, and unseen incumbents. Created-child H=2 eventually passed after both
structural and compositional-prompt repairs.

However, the broad public Simple-SWE factorized bridge stopped 2/7 attempted
because prompt evidence could not uniquely resolve leading structural tiers in
multiple families. That result retired further H=2 prompt-pattern patching.

Reference:

- [`public_factorized_simple_swe_train_bridge_v1.json`](../../experiments/r3_pilot/public_factorized_simple_swe_train_bridge_v1.json)

The replacement direction became bottom-up typed semantic synthesis.

## 6. Bottom-up typed semantic synthesis

A frozen bottom-up synthesizer using only typed signatures and four visible I/O
rows passed the first broad 30-task TRAIN gate:

- 10/10 scalar offset;
- 10/10 `List.map(add)`;
- 10/10 `List.map(abs)`;
- all 960 property cases excluded from model selection passed in the sandbox.

This established a strong **programming-by-examples** mechanism, not prompt-only
coding or domain-general learning.

Reference:

- [`public_typed_semantic_simple_swe_train_bridge_v1.json`](../../experiments/r3_pilot/public_typed_semantic_simple_swe_train_bridge_v1.json)

The next six-family TRAIN gate stopped after 13/60 attempts and 6 passes. It
localized three distinct issues:

1. complete-tier declaration overflow on scalar clamp;
2. retained bound-context diversity overflow on map-count;
3. four-row observational aliasing (`sum(x)` versus the intended clamp-sum).

The spent cohort is terminal and must not be retried or tuned.

References:

- [`public-typed-semantic-six-family-train-gate-v1-verdict.md`](public-typed-semantic-six-family-train-gate-v1-verdict.md)
- [`public_typed_semantic_six_family_train_gate_v1.json`](../../experiments/r3_pilot/public_typed_semantic_six_family_train_gate_v1.json)

## 7. Compact outputs and packed semantic families

The dense 3,284-cell stored output representation was replaced by a compact
winner-plus-digest representation while preserving physical bank identities and
race results. All 44 frozen representation/runtime gates passed.

Reference:

- [`compact-typed-output-encoding.md`](compact-typed-output-encoding.md)

The semantic frontier then moved to packed operator families, typed slots, and
physical semantic equivalence classes so concrete programs are materialized only
after a physical semantic winner exists.

Milestones:

- packed scalar V3 closes cost-5 clamp selection under the unchanged 30k work
  budget;
- packed full-typed slice composes `List[Int] -> Int` sum and both less-than to
  count orientations;
- cost-9 / larger-grammar scaling remains an active engineering boundary.

The generic typed breadth line then preserved scalar, List, and String routes
while adding a signature-only `List[Int] -> Int` construction. Its first fixed
six-row canonical panel selected three rows, kept scalar nonnegative and
clamp-map-sum explicitly ambiguous, and stopped map-count at the 30,000-unit
selected-proof bound. Generic V4 factorization later selected one self-posed
19-literal map-count proof at 16,436 units, but the separately named immutable
canonical V3 panel remained 3/6: the same two ambiguities survived and
map-count again reached 30,000. This localized the next seam to
prompt-authorized recipe/domain slicing before proof materialization, not a cap
increase, spent-panel retry, or host tie-break.

Current source:

- [`typed_semantic_packed_frontier_v3.py`](../../experiments/r3_pilot/typed_semantic_packed_frontier_v3.py)
- [`typed_semantic_packed_frontier_full_typed_v1.py`](../../experiments/r3_pilot/typed_semantic_packed_frontier_full_typed_v1.py)
- [`typed_semantic_packed_generic_train_breadth_v4.py`](../../experiments/r3_pilot/typed_semantic_packed_generic_train_breadth_v4.py)
- [`packed_generic_typed_canonical_train_freeze_v3.py`](../../experiments/r3_pilot/packed_generic_typed_canonical_train_freeze_v3.py)

## 8. Prompt-factor learning lineage

The original canonical TRAIN prompt manifest stopped before learning because 90
prompt appearances collapsed to only 42 distinct role-normalized physical
inputs and several factors lacked sufficient independent support.

Reference:

- [`public-train-prompt-factor-manifest-v1-stop.md`](public-train-prompt-factor-manifest-v1-stop.md)

Self-posed development then proceeded through several explicit generations:

- V2 showed small add/subtract and nine-factor atomic learning could work;
- same-union multi-factor composition failed;
- V3 local-span atomic generalization stopped at 40/54;
- V4 exclusive WTA improved to 44/54 but exposed shared-feature credit errors;
- V5 teacher-gated branch-local credit passed 54/54 atomic replay, 18/18
  two-factor compositions, and 21/21 compositor controls with causal teacher-
  off/cell-off/permuted-label negatives.

Current source:

- [`teacher_gated_factor_workspace_v5.py`](../../experiments/r3_pilot/teacher_gated_factor_workspace_v5.py)
- [`teacher_gated_factor_workspace_v5_protocol.py`](../../experiments/r3_pilot/teacher_gated_factor_workspace_v5_protocol.py)

Canonical TRAIN structural selection then exposed a separate representation
lineage. V1 stopped before learning because the available annotations had no
exact structural relay targets; a compiler-owned data capsule supplied abstract
targets without executable programs. V2 learned only 16/24 FIT identities: its
whole-target trace aliased identities and its FIT-only registry could not express
unseen atoms. V3 replaced whole-target heads with channel-local compositional
atoms and ordered transitions. Its learner passed 138/138 self-posed lessons and
six 0/138 causal controls, froze checkpoint
`7cb126fa194a96a26072f59d326cdb1a7c32cffa06bd5d07e27e28df8101720a`,
then stopped atomically at 19/24 canonical FIT projections. All five unsupported
identities required only `nesting-relation|depth|depth|4`; canonical teacher
updates and heldout reads/replays remained zero. The terminal V3 result is banked
at commit `582de79`; its successor must remove absolute depth rather than extend
the spent depth vocabulary.

The V4 protocol did exactly that: it replaced absolute depth with
occurrence-addressed ordered immediate parent-child edges, froze 103 atoms and
24 transitions over 110 fresh self-posed lessons, and admitted 24/24 FIT
identities without changing the six canonical structural classes. Its fresh
learner learned 110/110 self-posed lessons while all six controls stayed at
0/110, then used one authorized FIT teacher pass and replayed 24/24 identities
and 357/357 occurrences before physical STOP. Checkpoint
`e81336c6901edac0aada25c4ad89857686008fb276b94f705944ff9962a67d69`
is banked at commit `561a581`; all heldout counters are zero.

An integration audit then found that importing the learner reran training and
that the learned arrays were not available as a static consumer artifact. Commit
`77d341f` publishes a side-effect-free authenticated capsule and teacher-free
`infer_prompt(...)` seam. It reconstructs the exact checkpoint and banked actual
winners without importing learner, annotation, manifest, or target protocol
modules during normal inference.

Current canonical prompt source:

- [`public_train_prompt_selection_v4.py`](../../experiments/r3_pilot/public_train_prompt_selection_v4.py)
- [`public_train_prompt_selection_v4_protocol.py`](../../experiments/r3_pilot/public_train_prompt_selection_v4_protocol.py)
- [`public_train_prompt_selection_v4_static_checkpoint_capsule_v1.py`](../../experiments/r3_pilot/public_train_prompt_selection_v4_static_checkpoint_capsule_v1.py)

## 9. Joint prompt-factor + semantic selection

The first frozen self-posed joint matrix now connects learned V5 factor relays
to generic packed-family admission, then uses visible-I/O physical races to
resolve exact semantics before selected-proof materialization.

It passed 3/3 treatment rows and 4/4 controls. This proves the **joint seam** only;
it does not establish canonical TRAIN prompt learning, family-disjoint transfer,
or domain generality.

The three consumer mechanisms now exist separately. Commit `ce838bb` publishes
the exact factor V4 arrays as a side-effect-free static capsule with target-free
raw-span inference. Commit `41cc0a5` publishes a self-posed private
pre-materialization plan-slice receipt; its 19-literal map-count treatment
selects one proof at 26,073/30,000 relation units only after the normal physical
winner agrees with the authorized e-class. Its caller-created structural
certificate is not learned canonical authority.

The first canonical actual-record joint later reached a banked exact-object
runner. Its V1 spend stopped before semantic work because the protocol counted
70 span records while the producer normalized them to 69 unique assemblies:
the authentic candidate surface was 554 rather than 562. A fresh V2 protocol
froze the normalized geometry before visible-I/O pairing and selected two new
source-order appearances. Its separate batch/static/converter runner was banked
at `db01d6d` and executed once at `1722567`. Both producer-off rows passed at
554 candidates. Gain-one source selection on static-cells-off row 0 then failed
before static inference because no downstream certificate was present. Source
logic narrows that to zero selected events or rejection by the host 8/9 event-
shape gate, but the terminal artifact did not retain enough receipt data to
choose between them. V2 ended before static accounting, converter, desired,
proof, or normal work. Both V1 and V2 are terminal and immutable; each failure
localized a causal execution contract rather than a semantic result.

In parallel, the abstraction lane learned relative literal roles (`2f6637f`),
relative ordered operands (`50adc0a`), and residual-singleton routing into one
additive hole (`b181ec3`) from packed consequences. These mechanisms transfer
unseen payloads and physical order/position changes, but the program topology,
typed decoder, reducer, and destination holes remain explicit scaffolding.

Reference:

- [`joint_prompt_factor_packed_full_typed_v1.py`](../../experiments/r3_pilot/joint_prompt_factor_packed_full_typed_v1.py)

## 10. Consequence abstractions, terminal V3, reusable schemas, and scratch reduction

The adjacent self-posed abstraction lane continued beyond relative literal role,
ordered operand, and residual-singleton routing. Commit `aa91fa2` used append
consequences to learn whether output deltas follow predicate spikes or appended
payload, transferring anonymous count versus raw-sum reducer authority to fresh
prompts, physical permutations, and a same-prompt law flip. It added five cells
(3,449 total); the composed four-layer stack owns 24 trainable bytes. The
predicate/addition/reducer implementations and compiler remain scaffolds.

The canonical joint lane used commit `6848ad7` to freeze a separately named V3
exclusive one-shot runner with dynamic-`n` accounting. Its immutable terminal
execution at `b283053` selected authentic source sets of 8 and 7 events, but
both static-off and static-on returned `STOP_ZERO` on both rows. It consumed
67,380 fitted GLIF steps and opened zero plan builds, desired reads, proofs, or
sealed inputs. The cohort is permanently spent and must never be retried. This
is a same-FIT static-event failure localization, not a semantic or capability
result.

Commit `2837f1d` independently banked both procedural admission optimizations:
an ascending step-local fresh-GID run plus ledger union merge, and canonical
staged append/last-entry repeat without a production B-tree or redundant sort.
Focused synthetic medians measured 30.96x and 3.34x respectively while raw
state, counters, overflow behavior, API, and scientific accounting stayed
exact. Final source SHA-256 is
`3016fd0a44d891c8f2b8b161f5c2a011713e70cfd0848f0585442f969927958e`.
It does not change Path-2 evidence, neural work, or capability.

Commits `c587c32`, `2b709eb`, `6b6ec0e`, and `2e49f0a` successively removed the
duplicate interval-ledger buffer, phase-aliased the full-capacity GID scratch,
reused the current-step spike suffix, and packed the remaining interval flags
into three interleaved u64 bitplanes. The representative payload fell from
8,009,648 to 6,330,192 bytes; the exact reclaimed capacity is
`10D + 16R + 96 - 24*ceil(D/64)` bytes. Focused parity fixtures preserved raw
dynamics, queues, spikes, counters, failure boundaries, event order, and
accounting. This is implementation evidence only, not a 100-billion-cell
residency result.

Commit `7fa6231` then removed the separate staged-cell vector by phase-aliasing
compact staged records into the already-reserved event-work buffer. It reclaims
exactly `72*T + 24` bytes, preserves the public ABI/checkpoint record layout and
scientific ordering, and passes the focused release fixture. This is another
private scratch-memory result, not physical 100-billion-cell feasibility.

Commit `cf1cddb` then stopped full-touch stepping from reserving the
event-proportional-only `dormant_values` scratch. It saves exactly `8R` bytes in
full-touch mode while preserving event-proportional behavior and all public,
checkpoint, ordering, counter, capacity, and scientific semantics.

Commit `61b345f` then composed the append-effect stack with predictive
segmentation through only a learned two-cell `NEW/CONTINUE -> OPEN/EXTEND`
router. The 69,208-cell, 308-byte composite passed four fresh self-posed
`List[Int] -> Int` transfers: count, raw sum, representation/event permutation,
and a same-prompt output-law flip. Cue-off, router-cell-off, and append-effect-
off controls stopped causally. Host prompt/probe grouping, fixed byte framing,
interval cues, predicate/addition/reducer code, typed decoding, and compilation
remain explicit, so the claim is narrow composition rather than parsing,
arbitrary algorithms, public transfer, or coding capability.

Commit `17c891f` then learned two content-addressed causal segment schemas. Its
A/B/A-prime/fresh-B-prime assignments are `(0,1,0,1)`; four fresh transfers each
produce one exact proof, while prototype/schema/comparator/action-map controls
stop. The five-cell, 274-byte increment never reads the upstream selected action,
but probe grouping, consequence validation, reducer scaffolds, and compilation
remain explicit.

Commit `b5be799` independently learned anonymous variable-length occurrence
bundles. A label-free chain/fanout A/B/A-prime stream assigns `(0,1,0)`; fresh
lengths and opaque relabeling issue exact occurrence-graph certificates. The
19-cell, 178-byte increment stops under mixed topology and replay/relation/action
ablations. Host carrier allocation, endpoint framing, graph actions, and
consequence projection remain scaffolds.

Commit `064dd8d` then composed the exact occurrence certificate with the learned
content/action schema. A learned non-identity 2x2 uint8 E/H relay consumes only
the authenticated occurrence schema cell, wins one physical reducer-action
receipt, obtains a residual certificate and identity-issued plan-slice receipt,
and materializes exactly one count or raw-sum executable witness. Fresh C5/F4,
opaque relabel/reordered-record, and same-prompt topology-law-flip cases pass;
relation/relay/action/residual controls stop before packed construction. The
69,232-cell composite adds zero functional cells and eight learned bytes, with
5,580 bridge-only fit steps and 1,860 READY bridge steps.

Commit `c75458a` then exposed the raw-neutral grouping limit: separately framed
relabeled chain/fanout transports group under one opaque interval, but
continuous streams with incompatible hidden cuts collide exactly
`CONTINUE=16 / NEW=16` at the first novel edge and stop. No boundary surrogate
is added.

Commit `54e5a5c` adds four completeness-by-recurrence coincidence cells and a
16-byte uint8 E/H decision race. Plasticity observes only the unique authentic
READY consequence of trying both banked actions. Fresh continuous streams infer
boundaries `(4,6)` and `(3)` and issue `(A,B,A)` / `(B,A)` schema sequences;
completion, recurrence, gate, router, and replay controls stop. The absent
incomplete+novel training feature remains an explicit abstention. Training is
still transport-separated and environmental stream start/end plus carrier,
projection, reducer, typed, and compiler scaffolds remain.

Commit `315f865` next added an exact issued-certificate continuation to the
executable bridge; certificate checkpoint identity is authenticated and
recognition is not rerun. Commit `8b797a4` composed that seam with continuous
grouping, producing ordered A/B/A count/sum/count and B/A sum/count witness
sequences with zero new cells/state and causal gate/relay stops.

Commit `ad6eb57` published a desired-free physical ABS-map-SUM root with zero
new cells/state. Commit `7954f29` then learned an eight-byte IDENTITY/ABS relay
that selects exactly one raw-SUM or ABS-map-SUM receipt across fresh and
relabelled self-posed tasks. Commit `151402a` separately learned an eight-byte
predicate PASS/FAIL to accumulator ADD/HOLD transition; an explicit physical
equality relation binds the tested item to the accumulated item, yielding one
fresh conditional-sum witness with causal stops.

Commit `e146761` next removed the caller-supplied raw/ABS label for a fixed
two-candidate slice. Both physical consequences are visible before a receipt;
the example race learns one choice and stops on ambiguity. Commit `29e8d71`
fixed the generic rootless `add(x,0)` identity publication that the first
complete threshold-bank attempt exposed. A fresh 131-candidate retry then
stopped before receipt at thresholds `-64` and `64`, where inherited generic
predicate-count expansion loses a supported Bool binder. Both draft banks were
removed. Commit `6089f3c` then replaced that inherited expansion for this
mechanism with a full-range publisher that constructs only the authorized
predicate-gated-fold topology; `-64`, `0`, `5`, and `64` all publish and retain
their causal gate stops. Mandatory preflight then exposed inherited generic
predicate-count breadth separately in raw SUM at item 64 and, after raw was
repaired, in ABS-map-SUM. Commit `63ed457` banks the minimal raw publisher;
commit `d9c0d39` banks the minimal ABS publisher. Each authenticates its full-
range slice before outputs with zero proof/materialization and retains its causal
STOP. Commit `4efa736` then banks all 131 desired-free candidates: raw SUM,
ABS-map-SUM, and each predicate threshold from `-64` through `64`. One physical
semantic-vector race selects across the complete authenticated bank; ambiguity
and no-match STOP with zero materialization, and focused observations uniquely
select raw, ABS, and threshold 5. This closes the self-posed complete-plan-bank
increment after four honest blocker localizations, without candidate pruning or
host consequences; learned reusable fragment discovery remains open. The spent
V3 cohort remains closed.

Commit `f86a1a1` then removes redundant work without changing that evidence.
The 129 predicate plans share one desired-free multi-literal frontier and one
topology enumeration, but retain independent singleton-domain certificates,
receipts, cumulative work roots, candidate identities, and duplicate semantic
vectors. The focused complete-bank run improves from about 161.28 to 48.05
seconds (3.36x), with relation work 67,338→10,609 and races 4,773→1,064 under
the unchanged cap. This is exact construction acceleration, not a new plan or
capability result.

Commit `c728412` opens that discovery increment with a two-program
content-addressed fragment library. Physical selection of the solved plans
precedes fragment exposure; 16 typed subtree occurrences become 13 E/H cells,
with three consolidations across two recurring cells. A fresh desired-free
frontier reactivates those cells, forms generic type/scope-compatible one-edit
relations, and selects a never-trained `abs(item) < 5` program in one physical
semantic race. The original equivalent-composition target remains multiple-
exact rather than being pruned, and library ablation removes every recipe. The
result is narrow within-domain compositional reuse under explicit typed edit
metadata and host visible-case execution, not parsing or arbitrary induction.

Commit `d9e95d5` learns and recurrently applies two relations above that library.
The second lesson initially exposed an honest identifiability collision: direct
gated ABS and outer ABS have identical ordinary outputs. A generic singleton
execution intervention observes whether the already learned donor callable is
inhibited by predicate rejection, then one physical semantic race selects the
causal placement without an operator/path answer label. Two pre/post relation
cells and a five-feature/five-action E/H policy add 12 cells and 4,150 bytes.
Fresh construction applies exactly two selected edits, commits both after effect
races, and only then uses desired output cells once to materialize the novel
ABS-map/ABS-threshold-5 program. The result is one self-posed recurrent two-edit
composition under explicit typed and execution scaffolds, not broad coding.

Commit `3890523` then exercises that policy through one real microrepository
boundary. An attested buggy source and visible input cases produce the
desired-free two-edit frontier; visible expected outputs are read only for one
terminal race, after which exactly one source is materialized, one file is
patched, and one repository test is run. The learned effect policy maps the
observed consequence to COMMIT or REVISE, and the latter restores the original
source. Zero-write/test causal controls and the forced-failure rollback pass.
This is the first retained end-to-end repository repair in this line, but only
for one self-posed typed task with explicit environment adapters.

Commit `e49d227` repeats the mechanism test on a disjoint `String -> String`
family. Selected-program executions, rather than host-recomputed answers, teach
two membership/order/cardinality/lineage relation cells from a generic cost-7
frontier. The String checkpoint imports none of the List/ABS relation arrays;
it reuses only the authenticated generic action-policy state. Two fresh
cell-selected edits compose decimal filtering and reversal, with one later
desired race and repository patch/test. The 1/1 focused fixture took about 76.5
seconds and retained ambiguity, gate, drift, order, and rollback controls. This
is one within-domain family transfer under substantial typed environment
scaffolding.

Commit `9552144` connects the retained deterministic raw-chunk bridge to the
actual V1 repository boundary. The adapter now freezes and attests one source
snapshot before visible case loading and raw recall; one fresh A+D stream
reactivates two distinct learned relation receipts, and the unchanged V1 repair
performs the sole desired read, selection, materialization, patch, test, and
COMMIT/REVISE race. Preparation adds no learned storage and retains no raw
stream. The focused fixture passed 1/1 in 216.20 seconds with ambiguity,
unknown, no-match, source-drift, and exact failure-rollback controls. This closes
one raw deterministic motif→repository action loop, not language understanding
or general software engineering.

Commit `6010131` turns one real repository failure into a second physical plan.
Two source-free String candidate records are exact on fresh visible palindrome
cases. Receipt support makes the two-relation candidate win a compiled
three-cell GLIF race; a fresh non-palindrome repository consequence makes that
patch fail. The existing learned policy selects REVISE, the adapter restores
the exact original bytes, and a bounded +2 inhibitory update makes the
one-relation candidate win the next race and pass. Candidate order/remapping,
gate, forged-input, drift, repeated-winner, and two-failure controls preserve
the result. The controller adds 86 checkpoint-array bytes and 3,600 candidate
cell-steps; its 1/1 focused run took about 58.9 seconds. This is autonomous
correction for one binary-feedback two-candidate repair, not failure diagnosis
or broad continual coding.

Commit `37dbb9e` makes those real committed action consequences replay-free and
continual. After one String policy fit, a first Harbor episode fails then passes,
a disjoint Cedar episode learns the opposite successful action, a relocated and
renamed Harbor repository recalls its learned action in one test, and a later
Cedar episode fails its old choice before revising to the alternate. Immutable
successors consume only new authenticated action lessons: the intermediate
Cedar contradiction stops, the next ordered lesson revises it, earlier Harbor
evidence stays exact, and parent replay/mutation/raw retention remain zero. The
sole focused run took 54.510913 seconds and made six selected
materializations/tests, zero unselected ones, 10,800 memory cell-steps, eight
trainable bytes, and 1,024 checkpoint-array bytes. Policy root
`7eb12ff4bbf07d97effecb2744860fa9783c7d7ec144ec05264bcac0dad5ac78`
leads to final memory root
`c000a7d44de33d224ec69a4fb445a556ebb9f7a59bea9d030ab2037041f74246`.
The Harbor/Cedar comments are arbitrary categorical visible-source cues rather
than meanings learned from language. This banks exact recurring-source action
preference and revision, not semantic context, language understanding, or broad
continual coding.

Commit `76c23d3` generalizes the action-memory seam across separately learned
String and recurrent `List[Int] -> Int` policies. A neutral authenticated bank
owns exactly two frontier-bound candidate records and stable action identities
derived from the policy checkpoint plus ordered nonempty relation receipts. The
common controller now depends on a repository-action protocol, delegating typed
visible-output encoding, selected-only materialization, and the real learned
consequence race to each frontier. The List mechanism keeps root/final semantic
terminals separate from two nonempty learned-prefix repository terminals.
Committed result roots include the ordered lesson-body cohort, and every lesson
binds back to the controller trajectory and result.

Its A-B-A'-B' gate froze all frontiers before hidden fixtures: String A passed
with support 2; List B failed support 2, rolled back, and passed with support 1;
then relocated/renamed fresh String A' and List B' each recalled the corresponding
action in one attempt. Five selected sources were materialized and tested, and
zero unselected sources were. Candidate reversal, unknown typed bank, recall and
candidate gates, and relocation/renaming controls either preserved the choice or
stopped preaction. The single flat immutable memory contains two policies, four
actions, two exact contexts, and three provenance rows in 16 trainable bytes and
768 checkpoint-array bytes, with no replay, raw retention, or parent mutation.
The heterogeneous fixture and controller regression both passed after audited
fixes. Reusing the controller's authenticated source projection removes one
`P(n)=49n+24` projection; modeled successful recall at `n=256` falls from 28,736
to 16,168 cell-steps, about 44%. The List B hidden `[-2]` case overlaps prefit
examples, so this banks exact-source typed cross-policy action retrieval only,
not unseen semantic consequence learning, broad coding, SWE, or benchmark
evidence.

Commit `69cb0ae` implements the planned lexical source-factor route above exact
episodic recall. A fixed Python tokenizer maps at most 256 source bytes and 64
normalized semantic tokens into at most 24 deduplicated adjacent trigrams, each
projected through the generic common-event cells. One immutable replay-free
composite advances its exact child and factor rows from the same authenticated
controller consequence cohort. Two distinct source roots plus a unique
factor-local physical action win are required for admission; exact recall stays
authoritative, only exact UNKNOWN falls through, and factor gate, unknown,
ambiguity, and conflict STOP.

The one-policy focused gate learned from four training sources and six real
consequence lessons. Sixteen factor rows produced seven admissions: five
reverse-only factors select support 2, two copy-only factors select support 1,
and nine common factors remain unadmitted. Two byte-new sources each transferred
in one selected materialization/test with zero unselected work. Candidate
reversal preserved both choices; comment/string-only and neither controls were
UNKNOWN, both live patterns CONFLICTed, factor-gate ablation stopped before
lexical projection, duplicate replay was rejected, and the 256-byte preflight
bounded the observation. Cumulative factor/exact/total learning was
151,200/18,000/169,200 cell-steps; the composite stores 96 trainable and 3,530
checkpoint-array bytes. Fresh recalls cost 132,880 and 126,232 cell-steps, and
the cap-level maximum is 432,352 under the 500,000-step bound.

This banks fixed-tokenizer lexical motif association only. The byte-distinct
variants are not independent repository/task families, much novelty is removed
by normalization, the discriminating motifs occur in repair-irrelevant helpers,
and only one policy is tested. No source semantics, causal role/dataflow, broad
coding, SWE, or benchmark claim follows. Public DEV, canonical M0, held-out/C1,
G0 adjudication, Path-2 evaluation, and protected surfaces stayed sealed. The
next falsification gate should distinguish the same motif in causally relevant
versus irrelevant live code and transfer an equivalent behavior across different
syntax before a stronger abstraction claim.

Commits `a094653`, `531c6b2`, and `ad39f18` execute that next bounded
falsification step. An answer-blind Python adapter emits a complete rooted event
graph and a neutral receipt binds the source, graph, producer schema, and
producer attestation. A generic common-event-cell relay projects anonymous paths
into authenticated structural role factors. The new repository-action V2 facade
then orders exact episodic, structural-role, and lexical recall without importing
the Python adapter. The prior factorized exact-plus-lexical checkpoint is now a
private child rather than a compatibility API, and full V2 experience roots are
retained in frozen per-source provenance.

The four-source one-policy pilot saturated all eight role rows. Four selected
source-to-return-boundary factors admitted with independent support and
transferred opposite actions to fresh reverse and identity sources; four
lexically similar decoys accumulated conflicting evidence and did not admit.
Lexical evidence was non-decisive. Receipt splicing, exact-first precedence,
role/lexical gates, unknown/conflict, candidate-order, immutability, and forged
status/tier/work controls all preserved the boundary. The composite is capped at
192 trainable and 7,352 checkpoint-array bytes under 384 KiB transient,
512,000-cell-step recall, and 8,192-read caps.

This banks only authenticated, bounded rooted source-to-sink provenance
association and transfer. It is not general role discovery, learned causal
semantics, broad coding, SWE, benchmark, or autonomous-software evidence. Public
DEV, canonical M0, held-out/C1, G0 adjudication, Path-2 evaluation, and protected
surfaces remained sealed. The next gate must hold the V2 checkpoint frozen while
testing family-disjoint surface, packaging, decoy-context, and role-counterfactual
variants before any broader coding claim.

Commit `984cf4442c081d49a6024feeb35610e80c1a5be8` executes that public gate
without a cognitive change. One answer-blind harness first constructs all
source-only observations, then stages the hidden oracle solely for the selected
repository test. Its literal 12-case manifest root is
`f528a0d77aa173ec24d7fca650bbd3dc099b9f6513d2496dcad1f4c5b18b8198`.
Six reverse/identity pairs cover surface, syntax-packaging, the adapter's
length-5/6 parity probe, graph-isomorphic syntax, decoy context, and a returned-
role counterfactual. Pair mates are six rows apart, both action strata have
selected-occurrence counts `{0: 2, 1: 2, 2: 1, 3: 1}`, and six reversed
candidate orders are balanced 3/3.

All 12 byte-new positives recalled at the structural-role tier and committed in
one attempt. Action roots split 6/6; no unselected source was materialized or
tested; the frozen memory and bank roots did not change. Unknown, mixed-role
conflict, structural-role gate, producer-receipt splice, and candidate-order
controls preserved the boundary. The focused command
`.\.venv\Scripts\python.exe -m pytest -q tests\test_public_role_transfer_mini_swe_v1.py --basetemp .pytest_tmp\stage4-speed-audit-public-v1`
exited zero in about 98 seconds. The final test strengthening that pins the two
occurrence `Counter` values was source/static-verified without rerunning the
98-second gate. Evidence SHA-256s are
`25b0249d5caa9198eb79af6a438811011a8bde2cbd641f03bd78041cb9f85e9d`
for the experiment and
`8d559cf504ac41d50f196f5837df31030f88767ad190a6dbb90aff14b48def9e`
for the final test.

This advances the evidence from a four-source fixture to fresh
single-function microrepository structural-role transfer. It does not establish
true multi-file, package, or module transfer: `syntax_packaging` names only an
expression-level spelling. Lengths 5/6 are not broad distribution-shift
evidence, and the result is not LeetCode, external SWE, or a public benchmark
score.

In the memory lane, commit `78580cb` combined exact source episodes with a
sixteen-choice lane-local semantic transform view, retaining contradictions
while conflict and revision changed only the semantic route. It still refit the
exact view cumulatively. Commit `65e5808` added a flat replay-free exact delta
over one immutable V1 base at 96 bytes and one match cell per accepted episode.
Commit `900bc20` composed both successors: C and D each append eight exact rows
using only new lessons while semantic state conflicts then revises, and all old
episodes, provenance, parents, and unrelated state remain exact. This is not a
combined compression or corpus-retrieval claim.

Commit `b8f2baa` learned which view to retrieve for two recurring opaque query
contexts from public exact/semantic child output-cell consequences. Reversed
R1/R2 laws teach history-versus-current routing and the checkpoint transfers to
fresh R3; independent contradictions collide then revise, duplicate/dependent
provenance does not vote, and selected-child failure never falls back. The final
router adds 65 cells and 1,544 checkpoint-array bytes. The remaining boundary is
unseen-query-context abstraction rather than another exact-cue router.

Commit `e936323` crosses that boundary narrowly by learning evidence on separate
physical factors and composing it at recall. A and B are never trained together;
unordered A+B sums to semantic `(0,2)`, while singles tie tonic, physical
factor ambiguity/unknown STOP, and one then two contrary A lessons collide at
`(1,1)` then revise to exact `(2,1)`. Factor, router, and selected-child causal
stops remain explicit; child roots and replay-free advancement remain exact.
The result adds 65 cells and grows from 464 to 824 checkpoint-array bytes. The
next boundary is autonomous factor discovery from repeated raw context, not
another caller-framed factor tuple.

Commit `45a9a94` crosses that next boundary for fixed, externally presegmented
one- and two-byte contexts. It hashes each authenticated physical position-
value-binding assembly into the existing 12-lane factor geometry, admits a
branch only after two independent accepted provenance votes plus a unique
physical support/route win, and later equality-races raw branches before
invoking the unchanged factor router. Of six initial parent factors only
recurring A/B are admitted, so never-fitted raw `ab` composes to semantic
`(0,4)` without a caller factor tuple. Projection/plasticity controls admit
nothing; two contrary A observations collide at `(2,2)`, a third revises to
exact `(3,2)`, and duplicate replay adds no vote. The composite owns 75 added
cells and 2,208 checkpoint-array bytes. The remaining boundary is learned
variable-length chunking/segmentation rather than a larger fixed raw frame.

Commit `b08e373` crosses that boundary narrowly for deterministic opaque motifs.
Continuous physical byte-value assemblies carry no internal spans or position
authority. Predictor match cells feed explicit CONTINUE/CUT_REPLAY currents;
continuity also inhibits CUT, so simultaneous evidence resolves through physical
current cancellation rather than host priority. Unique causal actions train
four-feature/two-action E/H, cut triggers replay after reset, independently
supported recurrent traces are admitted, and their unordered composition drives
the existing revisable factor router. Never-fitted A+D cuts and selects semantic
`(0,4)`; ties stop, duplicates do not vote, contradictions collide/revise, and
opaque remapping/refit preserves the law. The composite adds 125 cells and
25,064 bytes. Fixed one-event lookahead and deterministic length-2–5 motifs
remain a much narrower claim than language segmentation.

Commit `7fef9a9` learns a small consequence bridge from those raw traces to the
two d9 relation cells. Physical candidate-intervention outputs, not relation
labels, train uint8 E/H rows with a two-independent-provenance threshold.
Fresh A+D segmentation activates both distinct receipts and the unchanged d9
policy performs exactly two commits before desired outputs appear. Unknown,
tied, duplicate, ablated, and opaque-remapped controls preserve the causal
boundary. The bridge stores 804 checkpoint-array bytes and no raw streams. This
closes raw deterministic motif→relation activation, not language understanding.

Commit `16ee379` opens a separate licensed-document acquisition path without
changing the coding-policy lane above. Acquire-only documentation groups project
lexical factors and declared symbols into common-cell assemblies; binary uint8
co-occurrence synapses retain the learned decision state, while authenticated
task and candidate projections meet only at recall. A physical common-cell race,
not a host argmax, authorizes at most one candidate materialization. The frozen
heapq state contains four admitted chunks, two associated groups, 46 factor rows,
two symbols, and 84 association updates. It owns 92 trainable bytes and 832
provenance bytes within 3,412 checkpoint-array bytes. Training used 108,361
cell-steps; the scored recall used 38,790, of which 12,000 were physical-race
steps and ten were synaptic reads.

Commit `0a5783a` preregistered one human-curated fresh self-posed Python-docs
choice between `heapq.nlargest` and `heapq.nsmallest`, with no prior scored
recall, materialization, or candidate execution. Commit `972a76f` preserves the
first and only result. The frozen memory selected `heapq.nlargest` by 5–4 learned
ticks and a 17–16 physical lane; its single authenticated materialization and
execution returned `[9, 7]`, `[-1, -3]`, and `[4, 4]` for 3/3 frozen oracle
matches. Candidate-order reversal reproduced the full recall root. Cell-off,
no-plasticity, signature-only, and 5–5 conflict controls each abstained through
their preregistered STOP, so all five controls passed. Result receipt
`019ac75d46f9dde4eaafa638aa4aa067350dc83ce04382476941451399bbe186`
binds the record; its exact file SHA-256 is
`220e8e866d7d46e763b7b843c8aee3086c7d448d7018d3ffe334837372a24c7d`.

The result supports only causal documentation-factor-to-API-action selection
for this one human-curated two-candidate setting. It does not support broad
coding, semantic reasoning, repository repair, LeetCode, SWE-bench, benchmark,
or general-intelligence claims. The next acquisition direction at that point
was a bounded, licensed Rust Book slice. Declared-symbol extraction remained a
human-ontology/annotation-assisted seam, and an authenticated Rust
candidate-handle adapter had to exist before a selected symbol could license
any Rust candidate execution claim.

Commits `cddaf51`, `c9b77ea`, `4b5738e`, `1e27a3a`, and `eb61f30` then opened
that Rust lane: a pinned Rust Book corpus is consolidated with provenance,
projected across all concept cards, and fitted into authenticated
knowledge-action associations. Commit `17321c6` binds two pre-existing Rust
2021 sources—channel delivery with sorted duplicate-preserving output and a
mutex-protected total—to opaque action handles. Commit `4d4ab22` adds the
consume-once selected-action executor with bounded toolchain reads, dependency
receipts, Windows Job Object wall/CPU/memory/process/output limits, exactly one
compile, and exactly one binary run.

Commit `2d3fb3a` freezes the fresh replacement Rust gate. Its public batch
commitments are payload
`ba9b1ed5bc532a5e792ee4379e9eaa296f7f268a7dd2d70f595770111502f608`
and file
`7ee9bbaba3e0d4bd95447cbeb1886eff2ad67ad9dd7e81e37703b2571fa2df54`;
the sealed oracle commitments are payload
`7abb4f962ff886a36bd45225070fbb2f092edbf4ae32c6ca9701e573d9d109b0`
and file
`46539d4e2e800f564b2c5207ca0aafe8f3c5d420692272b55c2686598f075be7`.
Commit `bec231b` banks READY_UNSPENT manifest root
`072a94efd105d780ff708c0c482936aefef109c5ecc77bd5542553702dc95c0f`;
all six evaluation counters are zero.

Commit `4c8ead7` preserves the only spent evaluation as PASS. The ten-item
control roster contains nine physical recalls and a pre-physical
bound-memory-splice rejection. Primary channel, mutex contrast, source-order
reversal, cell-off, plasticity-off, shared-factor tie, unknown-factor,
single-channel, and single-mutex outcomes all matched the frozen contract.
Only the primary channel winner materialized. One authenticated executor call,
one compile, one binary run over four cases, and one post-execution oracle read
occurred; all four output rows matched. Terminal result root
`2f7b49d6e227263641fd840ac573b4a0bb59246d09ef472e2a704f260ef0dcc3`
and exact file SHA-256
`0471a2ad3949449bc6453b8346bc7dc8abc93890936985b854f68d85897f538f`
bind the record.

Two chronology incidents are retained explicitly. During protocol development,
a generic JSON-validity command opened an earlier uncommitted oracle before any
learner or gate work; it printed only `json-ok`. That cohort never reached
READY or spend. Those bytes and their public batch were retired, and the scored
replacement oracle above was newly sealed without being reopened before
authenticated execution. After READY was
committed, the first shell launch used the runner file path and exited on
`ModuleNotFoundError` while resolving the protocol import, before `main()`, lock
acquisition, spend, live preparation, recall, materialization, compile/run,
oracle access, or result publication. The committed manifest remained READY,
while spend, result, and owner-lock checks confirmed no irreversible attempt
artifact existed. The module entry point then performed the one actual attempt;
this was a pre-spend launch correction, not a scored retry.

The result supports only fresh lexical Rust-Book-documentation-conditioned
dispatch between two existing actions followed by one trusted-host compile/run.
It does not establish Rust synthesis, semantic code understanding, debugging,
transfer, continual coding, hostile sandbox containment, broad intelligence, or
comparison with Gemma/Opus. The next R4 question is whether documentation,
execution feedback, and learned action abstractions can accumulate across
multiple disjoint coding tasks without tuning on this spent cohort.

Commit `384aba4` begins that question at a synthetic mechanism boundary. The
new continual documentation-action memory consumes one newly sealed lesson per
immutable successor; it neither replays prior lessons nor mutates a parent.
Successful and failed public consequences add excitatory/inhibitory ticks only
on exact lexical factors shared by the authenticated task projection and
documentation-evidence projection. Per-lesson provenance is retained while raw
task text and output tuples are discarded from the checkpoint. Authentication
reconstructs the learned matrices from provenance, and two distinct successful
task projections plus two distinct task-source roots are required before an
action may enter the common-cell physical race.

The deterministic A1→B1→A2→B2 fixture interleaves two candidate-provided opaque
actions and then recalls each from a fresh task projection without refitting.
The final state contains four lessons, two factors, two promoted actions, eight
trainable bytes, 104 total checkpoint-array bytes, and 4,903 cumulative
learning cell-steps at checkpoint
`1bb914c1d106483e75a59aebfac4f2394c6773fa3961eb7d239a0d47943baef3`.
Unknown, unpromoted, tied, cell-off, and documentation-off probes physically
STOP; failed fresh evidence does not promote; a failed unseen action and
candidate identity splice are rejected; and a re-rooted selected-candidate
receipt cannot change the physical winner. Four new and ten adjacent tests pass.

This is not an adjudicative gate and did not materialize or execute either
candidate. It supports only replay-free continual lexical association and
recall over synthetic adapter-provided evidence. Authenticated Rust-corpus
evidence, selected-source custody, execution, semantic acquisition, synthesis,
debugging, and SWE transfer remain for fresh experiments.

Commit `0274178` added the next environmental seam: a promoted Rust action vault
rooted in one immutable continual checkpoint. The vault authenticates the full
candidate catalogue, promoted subset, promotion provenance, candidate-provided
Rust sources, explicit authority, selected release, and one-shot callback. It
does not receive task outcomes or choose which action should win.

Protocol commit `f586e990f9441dacb5d51437dd99d6d5e57adfa5` then froze a fresh
two-action iterator cohort. Result commit `ba7aaa9` records its only run at PASS
root `494e3a4696f2464cc4ed7d4ba9defd4c614d2fc13a5afc921a7a77741d9ef37c`.
One continual memory consumed four new host-supplied successful lessons without
replay or parent mutation, promoted both actions under the independent
task/source quorum, and made five physical recalls. Three controls stopped.
Two fresh exact-factor-overlap instances each released, compiled, and ran one
selected source; both matched an oracle first invoked after both executions.
This closed action custody and execution but deliberately did not test learning
from the executions' consequences.

Commit `a364d0a` froze the public-development C0 successor after primary-source
research and three interface designs converged on a neutral intervention plus
the existing memory/vault/executor stack. The intervention binds two
candidate-provided Rust actions, reverse (A) and ascending sort (B), without
receiving task objects/text, expected-output tuples, or memory state; it accepts
only opaque rooted intervention identities. Five authenticated
successful executions teach A on `kavun`, A on `kavun pelto`, B twice on
`pelto`, and A on `zimra`, producing an A3/B2 score on `kavun pelto`.

The adaptation reuses one exact task, instance, source, evidence projection,
batch, and expected public consequence for both attempts. A first executes and
fails. Advancing with plasticity disabled returns the identical parent and a
second recall still selects A. The actual immutable successor records the fresh
failure execution/feedback/lesson roots and adds two A-inhibitory ticks, so B
reraces over A at 2 versus 1. B then executes on the same batch, passes under a
fresh selection/execution/feedback/lesson chain, and its successor adds support.
Frozen evaluation uses two new task instances: exact `kavun pelto` factors select
B and `zimra` retains A; both compile/run before the public oracle callback and
neither updates memory.

The trusted result is
`PASS_CONTINUAL_RUST_FEEDBACK_REVISION_COHORT_V1` at root
`692c8545408de2d71b803aaec39f9e392b909f35ccf3f0b524b23a97da9814a6`.
The trained→failed→final roots are
`9e1506a7715337a7d7750b0c99ae038aa33637e374fd7cc7f99c52e13c2629ee` →
`8ba7d60db52b6b94652bf722735f89ccae650cfc945e14a45f4c352e4c13ec20` →
`8ea2f8c0f5077d79e438a97f8da564596699c95dc75336f05450251e7f7e27bd`.
Both evaluation cells matched. Work is exactly nine selected releases, compiles,
and runs; seven new lessons; zero unselected executions, evaluation updates,
parent mutations, or lesson replays; and one oracle callback after all nine
executions. The external transcript re-authenticates the full public spec,
task/evidence/batch/stdin commitments, intervention/vault catalogue and
authority chains, binding/release/source receipts, executions, feedback,
lineage, and oracle ordering.

This result establishes only public-development consequence-driven same-task
binary policy correction and persistence for two prebound sources under exact
lexical-factor reuse. The expected outcomes are public, and the action identity,
source implementation, and lexical evidence are host-provided. It is not source
synthesis, semantic understanding, diagnosis, generic debugging, paraphrase or
family transfer, SWE capability, or intelligence. The next experiment was C1:
learn and consolidate one generic parameterized token-edit relation from two
independent failed→successful episodes, then freeze it for a fresh C2 transfer
cohort.

Commit `f2de425` implemented and spent that public C1 cohort. The common learner
projects bounded Rust-like source into lexical token frames and races a fixed
24-member bank spanning six anchor kinds and two target/two donor directions.
Each episode seals exact failed/successful frames and authenticated public
consequences, while the immutable checkpoint retains only per-schema independent
source support and compact provenance roots. A normal one-source state tied the
tonic cell at score 2 and abstained; a separately labeled tonic-off control made
the underdetermination visible. Adding a second, independently rooted repair
made schema 3 the only member with support two and produced a normal `READY`.

Episode A changed the right operand of `reading > reading` to the local donor
`ceiling`; episode B used different names, literal, source text, material roots,
and `match` control syntax to change `temperature < temperature` to `floor`.
Their only shared fixed-bank explanation was comparison-anchor/right-target/
right-donor. The frozen fresh source instead used a `for` loop with
`measure >= measure`; the learned schema bound the new query-local donor
`boundary`—absent from both training frames—and proposed the one-token edit
`measure >= boundary`. Expected fresh edited source was used only in the audit;
the proposal itself was neither compiled nor executed.

An initial sandboxed discovery attempt stopped at `rustup which rustc` because
the sandbox emitted a path-canonicalization warning; it executed zero cohort
sources. The exact precommitted test was then run once on the trusted Windows
host and passed after exactly four selected source releases, compiles, and runs,
two FAIL and two PASS consequences, two advances, two parent-immutability checks,
and zero proposed-source executions. The spec root is
`3dedb36b3ae4c7a916e1626a32b598f9ed1ee7c23cde2b9a1bc10dabea3d6609`,
the learner-config root is
`84b60787deed505352885fad33abe58ba8b0d6c1d16cee688a7d19da0eb912a5`,
and promoted schema root is
`7cf44511bced68635e3c77a199542061b404b5861fe6a51c7c6ba88b01c2b014`.

The trusted test authenticated the full result in memory but exited without
serializing its result root or transcript. No second run was made to repair that
evidence-handling error. Instead, a weaker attestation at
`58b5a17957f8df0a7613b010d8e7007337f5a21c1c54a749ead746659ef44983`
records the exact known hashes, work, PASS outcome, and missing-root limitation.
Thus C1 supports replay-free selection/consolidation of one supplied lexical edit
schema and fresh query-local binding. It does not show learned tokenization or
ontology, semantic diagnosis, proposal correctness, C2 family transfer, SWE
capability, or intelligence. The immediate successor was a fresh C2 consequence
cohort that would not reuse or rerun the spent C1 evidence.

Preregistration commit `3316fb0026e85023f68802aa014ffedd7221079e` froze the C2
successor before execution. Its spec root was
`70bfdbc00bafb4e6a2d4d569dd74583d4dfafc104254ef2fb068f70fea09eafe`;
the frozen module and test SHA-256 values were
`906f9786b461015c10a56d408e50d6907c084522d7a68c5c25c6f7ca3719875f`
and `3d108f592a8309aa4fecda62f84154c575eeb3ebdb16be5dfcfafdda48ec7f52`.
The cohort was new and fresh rather than C1 checkpoint or receipt lineage. It
reconstructed schema 3 from two new content-root-distinct synthetic
failed→successful episodes, preserving one-source, cell-off, and schema-off
controls before the held-out executions.

The unedited held-out source compiled and produced baseline outputs
`((1, 1, 1, 1), (1, 1, 1))`, a public failure. Before either transfer execution
or oracle opening, the frozen checkpoint proposed exactly one edit: token 76
changed from `signal` to the query-local donor `threshold` at token 78. The
edited source root was
`ca96a9774be460daed557117e0b56fc334d29d709d9fac0a6f53003e3fd9be34`.
Those exact proposal bytes compiled and produced
`((0, 1, 1, 0), (1, 0, 1))`, a public pass. The authenticated work is exactly
four training and two transfer compiles/runs, zero retry, alternative, or
C1-source-reuse work, and one oracle opening after both transfer executions.

The sole reserved spend terminated
`PASS_C2_FROZEN_SCHEMA_EXECUTED_CONSEQUENCE_V1`. Its result root is
`afaee2199422b46de53abc78474203f0fbaee1d330cbceaddfc40bac1653d879`,
document root is
`ee5f68119b6ec84a241222c5fe74f5b7069e22cf5f5dae3b13b6e380205d05e1`,
and raw result-file SHA-256 is
`9a8cd9f7725b017758f9443b3c41366e1dcb415ead4ee43efa06bd5ce1bc7e87`.
Evidence commit `9e3a20d9dcd25868690136d6013da81acd0b0e65` banks the
terminal and its no-replay reservation.
The preceding reservation has authenticated root
`9a1e0d19712a8d400e19b5a5e49b60f91d5f443a7fc4af2a75557673d9830c26`
and raw SHA-256
`8614002947914fcf0aab8d8593f2773df97d04d2f4d0ce6e9b048b930977ae86`.
The frozen writer retained the reservation before external work, but its own
authenticated fields bind only the shared spec and exact destination; the
reservation does not independently commit the later result/document roots or
prove temporal ordering.

The transfer consequences were retained but did not advance the checkpoint;
therefore this is not consequence learning. It supports only replication of the
frozen lexical token-edit mechanism on two new content-root-distinct synthetic
episodes plus one held-out executed public consequence under exact V2 source
custody. It does not establish C1→C2 checkpoint-lineage continuation, learned
semantics, diagnosis, generic repair, protected transfer, SWE, or intelligence.
The authenticated terminal is authoritative. If interruption had left only the
reservation or a pending file, the outcome would be indeterminate—never PASS and
never authorization to rerun the cohort.

Commit `d9a3018` introduced common immutable action-outcome memory and a thin
token-edit consequence adapter. Preregistered C3 commit `9a4bbd3` froze spec
`18294356e4d71553a42603d2fc8ab48f8d94f232b269b5ae55515ab7de32fd2e`;
its sole spend terminated `ABORT_C3_TOKEN_EDIT_OUTCOME_REUSE_V1` before the first
compile or run because sandboxed compiler discovery emitted a
path-canonicalization warning. Commit `49a3365` retains abort root
`68de5bdf11ab9792e10a5fc027dd8d7e70f9ffaaacae371196159fa66d0111ea`,
document root
`db301d77e2fac7693087de709980c98d5e6e633df36966b393206174d6f0789b`,
and reservation root
`a3ea4e49b21c18c4bbca4c839e4263236b1908ad787be866fc17a571f619cb5c`.
The cohort was not rerun.

Fresh C3b commit `d731343` froze spec
`0905ff1099e38db828edf86135ba64c94b2e21ad63ea991db7c937ad0e6e9576`
after adding a pre-reservation compiler preflight. Its sole spend terminated
`PASS_C3B_TOKEN_EDIT_OUTCOME_REUSE_V1` at result root
`6344b4c32810a65423908da89c392182f68c0610ab2a0dbba22844db0b9099a5`,
document root
`8f71ad16c1f80c8b6c8e333510881eceb6b3997d2d51e3c3d2b9e010896f5db3`,
and reservation root
`46fdf844adb216f8ff7dcf396fc4cf0ab89f4e38406cb8df21de690ac5d735bd`.
The exact consequence vector was `(0, 1, 0, 1, 0, 1)`. One seed pair supplied
schema-3 support one; normal adaptation stopped, explicit exploration issued the
proposal, and its PASS advanced outcome memory once. On a fresh evaluation
query, the parent and outcome-gate-off successor stopped while the normal
successor was READY at score four; its baseline failed and proposal passed
without another update.

Exact work was six selected executions, compiles, and runs, two feedback
openings, one schema advance, one policy advance, and zero evaluation advances.
This is one fixed-bank consequence-reuse result under selected-only trusted
execution and caller-attested outputs—not learned tokenization, schema discovery,
semantics, diagnosis, generic debugging, protected transfer, SWE, or
intelligence.

Commits `0dd2006` and `11e29b6` then introduced the generic
failure-conditioned action-replan seam. One authenticated selected-action FAIL
advances durable inhibitory outcome evidence exactly once. A deterministic
ordinal-1 retry bound to that exact outcome and context may apply one transient
`-2` refractory pulse to the failed full action tuple and physically rerace the
otherwise unchanged candidates; the pulse does not alter durable credit.

Preregistered C4 commit `6727a51a01e0ce4d933402492f2d76ab2f46b55f` froze spec
`19e88590e6158f69e7c995b0af018aecc2973a14721394e5d74e762e819ffbc9`.
Its one-shot spend terminated `ABORT_C4_TOKEN_EDIT_FAILURE_REPLAN_V1` before
the first run. The first selected source compiled, but rustc emitted default
`path_statements` warnings from bare `floor;` and `ceiling;` statements, which
the warning-free executor correctly rejected. Commit `f2c524a` retains abort
root `5132dfec0d80f8fe3bb64247e744002941940d26918eb4c879abaad36b50b1f8`,
document root
`79435e597d2ecf7423b94a1228d5f41fb7beea5ef1511d8eb06d1c8052619f03`,
and reservation root
`206497f56bba6a06be5e02fbe2a96758a775b7e651474243dc79ae0171b78ed1`.
The cohort was not rerun.

Fresh C4b preregistration `7413af75243e974f2d018b533265923ce20ff317`
excluded all 30 previously spent source roots and froze spec
`ab3954e583d95b4975c495a02e0e38509b1681c2025239381031bfaf3216bd08`.
Its sole spend terminated `PASS_C4B_TOKEN_EDIT_FAILURE_REPLAN_V2` at result
root `bcbbc882256917bacccb492c03841813b351f7a2ccc56b000d1cf0fff31f3434`,
document root
`db75bf06aa14e5a31e64830e9e28ffcb5109b31f695f68cfa945ce307861659f`,
and reservation root
`c930744b2fcc7507c2f8dae26b2042e0592776b667849654f7ade44028d611b9`.
Raw result and reservation SHA-256 values were
`c64f22e8d2b915fc4a007a2663670d30164a45afa9a5842ee83f4e4bf16233ff`
and `acb96a244c4e10f6363a5078d8af6450cb4b6b8b833d65247fd711078aad1941`;
evidence commit `98b4339` retains both. The trusted rustc identity was
`aa7c4003101dc045a3689cf9f5c2c91bd0adedf19c9480f8282ce0c9b28efc84`.

The executed consequence cells were `(0, 1, 0, 1)`. Selected adaptation schema
A failed and wrote one durable H tick. The unchanged A/B tuple then scored 4/4
and stopped without refractory activation; the context-bound pulse made the
effective retry 2/4 and selected B. B passed and added one durable E tick,
leaving final A/B scores of 4/6. Parent checkpoint
`deb2bda9274ce1555a43f0886ec4121265b7160fb23022f4644ea0d77ddf255a`
selected A on a fresh source, while final checkpoint
`dc25c774e64b8320b7b8e660eba51f33bc5313f67b7e55dddf747c48fda5f690`
selected B; evaluation A failed, evaluation B passed, and evaluation advanced
nothing.

Exact work was four selections, selected executions, compiles, and runs; two
feedback openings at cumulative execution counts `[1, 4]`; two outcome
advances; one rerace; one combined gate-off control; and zero evaluation
advances. Both authorities independently recorded exact
execution/selection/release/spend/success counters of `2/2/2/2/2`. This banks
failure-conditioned reracing and fresh reuse inside one supplied two-schema
bank—not learned tokenization or schema discovery, failure diagnosis, inference
that the rival is correct, search or source synthesis, generic debugging,
protected transfer, SWE, or intelligence. The next mechanism question is
whether structured execution feedback can support reusable factor components
rather than only inhibit the failed action.

Commits `9c8fe0b` and `fe6500e` then introduced a generic factorized-feedback
hypothesis memory and a thin rustc JSON adapter. The adapter authenticates one
compile and source custody, uses each selected primary-span midpoint to emit
only error-code and relative source-half cues, and excludes messages, labels,
rendered text, and suggestions from cognition. C5 preregistration
`989c4b3ffac360bb8e4746166040926b1e4aa30e` froze spec
`52b62afa4df5f1c0d1f9e35bbfcd3872afc6fc74939e3e3be91a3f985b4d6d2c`
and 15 source roots fresh against 36 excluded prior roots.

Its sole spend terminated
`PASS_C5_RUSTC_DIAGNOSTIC_FACTOR_RECOMBINATION_V1` at result root
`3b3571b17ef5c5763ba8eda9ee4972334589f00e4cc4316b33cd90e79b0df1bd`,
document root
`e7fecae8479faf467b9e13af960bbe3fbe1ca50c478ecb83907d8261b9613f1f`,
and reservation root
`5fa4018d0c08b3102db2c8ae463d2e3cb2963fa419d53d5e24559fb176068ff8`.
Raw result and reservation SHA-256 values were
`86673903cd3467ec80f2e2e1a08f779914d75afc947088f1c6cde5b55709e547`
and `30312a9ee15281e67ff801dbbc3da665624d2be4a4250005d08699a386d2a63f`;
evidence commit `378abf3f3a3a06353f227c3e444ee03e923cff0d` retains both. The trusted
rustc identity was
`aa7c4003101dc045a3689cf9f5c2c91bd0adedf19c9480f8282ce0c9b28efc84`.

Normal adaptation stopped on the first `E0425`/first-half diagnostic. Explicit
TRAIN exploration selected its diagonal hypothesis, whose bound Rust source
compiled and passed, and one immutable memory advance followed. Normal routing
again stopped on the second `E0308`/second-half diagnostic; explicit exploration
selected the other diagonal hypothesis, it passed, and the second immutable
advance followed. Both plasticity-off and consequence-off controls preserved
their exact parent checkpoints.

On the fresh `E0425`/second-half diagnostic, both the empty parent and the
one-lesson checkpoint stopped. The two-lesson checkpoint scored the previously
unselected prebound cross hypothesis four and physically selected it;
outcome-gate-off and both single-axis
controls stopped. Its bound source alone was released, compiled once, ran once,
and returned the expected `((5, 15, 20),)` output. Evaluation performed no
memory advance.

Exact work was three diagnostic compiles, three selected candidate compiles and
runs, three feedback openings, two memory advances, zero evaluation advances,
and zero unselected releases. Each independent authority recorded exactly one
issue, spend, source release, execution attempt, and successful execution. This
banks one public TRAIN case of factor recombination under real compiler
diagnostics—not learned cues or localization, diagnosis, repair discovery,
source synthesis, generic debugging, protected transfer, SWE, or intelligence.
The next experiment still needed to remove a supplied abstraction by learning
reusable diagnostic components or their hypothesis bindings from causal
consequences.

Commit `a6db0190d20fb9be8c5429691a3fefaab65b461b` then added a generic
one-shot factorized text-edit materializer plus narrow selected-source executor
integration. Preparation freezes an original source and diagnostic, exact
factor-memory checkpoint, exploration policy, two supplied operator bindings,
and two supplied target markers. It authenticates marker declarations but does
not compose an edit or repaired source. Only an enabled decision from the exact
checkpoint can map its selected opaque components to one operation/target pair,
compose one LSP-shaped byte-range edit, seal one materialization/release, and
issue one source.

Fresh C6 preregistration `767f885ad12be5a67ea58be629ce1ba7476c9520`
froze spec
`b514ff417680b0a7238e705b9cfa110bf1b9c94c1b32f16a3ad084e64792d394`
and three original source roots disjoint from 51 excluded prior roots. Its sole
spend terminated `PASS_C6_RUSTC_DIAGNOSTIC_FACTOR_MATERIALIZATION_V1` at result
root `47bc6cba38eca01b8b61ce0c5086fd712d5090a7c01f95f7a071d55c5a9f471f`,
document root
`70edacee612c80268ee9b63f592a730f9a2043f4959842588f8331ff54226c33`,
and reservation root
`1e0124da9c5f31e5e60ca3b4f178a45ba367757619027183d28ac0f3ff9fd2e3`.
Raw result and reservation SHA-256 values were
`7aa2d8a8fe3232165ac5e7ad15e924b52464355b9656c01cf024a9a1492616c0`
and `d251bf7863de75e533d70e827a92fac09ef270266956a04b95aac107e85ac2c4`;
evidence commit `1ec0f17ef8cb14420c6884db55b504ec96003e6f` retains both.

Normal TRAIN routing stopped on the `E0425`/first-half and
`E0308`/second-half episodes. Explicit exploration selected H00 then H11;
their authorities were bound to the parent and first-successor checkpoints,
respectively, and each composed/executed one passing source before one memory
advance. On the fresh `E0425`/second-half evaluation, the parent and one-lesson
checkpoints stopped. The final checkpoint selected previously unselected H01;
its authority was bound to that checkpoint with exploration forbidden, then
composed and executed exactly one passing source without an evaluation advance.

Exact work was three diagnostic compiles, three post-selection
materializations/releases, three selected compiles/runs, three feedback
openings, two TRAIN advances, zero evaluation advances, and zero alternative
materializations. This removes the prebuilt repaired-source catalogue from C5,
but not the supplied cues/projection, factor ontology, operator semantics,
target markers, or output oracle. It is not learned localization, diagnosis,
edit discovery, generic source synthesis or debugging, protected transfer, SWE,
or intelligence. The next research step remains learning reusable diagnostic
components, operations, or bindings from causal consequences.

Commit `1f6afea1186cd0d43001db0560c4cde996534215` then introduced an
authenticated two-action text-edit catalogue. The factor memory's operator axis
now contains the catalogue's concrete DECLARE and REPLACE action roots rather
than opaque diagnostic-class components that a cohort-local table later maps to
operations. The catalogue factory alone supplies the root→implementation map to
the unchanged C6 materializer, preserving its exact selected-source executor
authority and the already banked C6 hashes.

Fresh C7 preregistration `ec5280c7f878eeabc361a067d7ab7f146c93e3c5`
froze spec
`7f90c6c393cc0a7a3a731672b7b24737c58621dfa892f8dff3745538a992939d`
and three original roots disjoint from 57 exclusions: all 51 earlier roots plus
C6's three originals and three materialized sources. Its sole spend terminated
`PASS_C7_RUSTC_DIAGNOSTIC_OPERATOR_ACTION_LEARNING_V1` at result root
`71b58d208f9edc373983c44e54d7febbfda115c342895f1ae9aacf02feb2631b`,
document root
`ee8456ab4e51a53ef14b06b6c6cef82260629b1ecbe0713bfc9c222ba86fa092`,
and reservation root
`5a69ebf3b5f7663c205d82ae86593e3984543edcbddde0969bcab27b0ba193de`.
Raw result and reservation SHA-256 values were
`46b69cb97fb33d54f268d79fc0d79080fae5736ba70c630ff60b05e17b0c92f6`
and `2225849ccaeb04475a4dbaee3438cbb134b4b46f1f23b1ba9b00cbed280b44c8`;
evidence commit `e1fead7d314580e79cbc88f362f2e0d0bb38191e` retains both.

Normal routing stopped before each TRAIN lesson. Explicit exploration selected
DECLARE×first-target and REPLACE×second-target, their selected sources each
passed public output feedback, and two immutable memory advances followed. On
the fresh `E0425`/second-half evaluation, the parent and one-lesson checkpoints
stopped; the final checkpoint normally selected the previously unexecuted
DECLARE×second-target combination. Its exploration-forbidden authority
materialized one fresh source, compiled and ran it once, and passed without an
evaluation advance. Outcome-off and both single-axis controls stopped.

Exact work was three diagnostic compiles, three materializations/releases,
three selected compiles/runs, three feedback openings, two TRAIN advances, zero
evaluation advances, and zero alternatives. This banks consequence-conditioned
selection of concrete authenticated action identities, not action discovery,
operation-semantic learning, target learning, cue discovery, learned cue-axis
routing, diagnosis, synthesis, debugging, transfer, SWE, or intelligence.
Established contextual/factored-bandit and reward-modulated eligibility
patterns point next to an unordered cue×action-axis×component association. Two
diagonal episodes do not identify that routing; the successor must use crossed
or full-grid evidence, or expand an axis if it also needs a held-out
composition.

Commit `2eae160c645b1e5fb0be996b7d1d4bac7c862508` froze the generic C8
mechanism before cohort construction: one unordered opaque cue set is uniformly
projected to both axes of the existing physical factor race, raw outcomes remain
in an authenticated outer lineage, and a new direct authority recomputes its
observation from the retained rustc diagnostic before selected-only
materialization. No C6/C7 frozen mechanism was modified.

Fresh C8 preregistration `6f97d4d3b4c8643dab5c55502586bc9430924787`
froze spec
`a6372e986cec76b04e9e9efd412cc6e3ae5c1ba862566801acf2f8c02d3f9fda`,
63 prior exclusions, six fresh originals/materializations, and a complete
H00→H11→H01→H10 grid. A sandboxed preflight first failed to resolve concrete
rustc and created neither reservation nor result. Read-only trusted-host
preflight then authenticated rustc 1.94.0 at identity
`aa7c4003101dc045a3689cf9f5c2c91bd0adedf19c9480f8282ce0c9b28efc84`;
the exact preregistered cohort was subsequently reserved and run once.

That sole spend terminated
`PASS_C8_RUSTC_DIAGNOSTIC_UNORDERED_CUE_BINDING_V1` at result root
`7aa8981b01609b1e74acc8878e26ba2ed210ccd8e6fa29bab86451323ad00c2a`,
document root
`38a80807cbb23349c69360da941ef826dab0ed0203e9c95f682ee46abd660009`,
and reservation root
`60382c21eee70c2c0999109ac536bd346159aad862ceb40f7c6d69319365b328`.
Raw result/reservation roots were
`90348ce1b2247a235ae61516a1af581c61afcf79f37fe055a046e913e91b4520`
and `3de04bc24c95af6c4dda34f268b0487bfaaba4b3753cb00782194b69451c798e`;
evidence commit `02a667c880b31946724df0bc5739f75f71a0cbf3` retains both.

The parent and diagonal checkpoints stopped on both crosses. After H01, the
three-lesson checkpoint selected that learned cell while the missing H10 cross
still stopped. The full checkpoint selected both fresh-source evaluations with
the frozen `12/8/8/4` score pattern. Both selected sources executed before one
combined oracle open and passed; all outcome-off/single-axis controls stopped.
The authenticated work was six diagnostics, six selected materializations,
releases, compiles, and runs, four TRAIN feedback opens, one evaluation oracle
open, four memory advances, no evaluation advance, and no alternative.

C8 removes the supplied cue-axis partition and establishes binding reuse from
an unordered supplied cue set. Full-grid training makes the binding claim
identifiable but deliberately forfeits a held-out-composition claim. The rustc
cue abstractions, action/target ontology, exploration answers, edit semantics,
markers, and outputs remain supplied. The next useful mechanism was to learn at
least one raw cue abstraction or target binding from causal data rather than
wrap another host-labelled factor.

Commit `d8b40775ae90fd289188464dddd008876452bcfc` next replaced the
explicit diagnostic-code cue with a generic rustc parent-message adapter. It
lowercases and splits ASCII-alphanumeric word unigrams, hashes the resulting
tokens, appends the existing source-region root, and owns exact attempt/stderr
custody. Code roots, children, rendered text, labels, suggestions, and edit data
are excluded. Commit `287d7b322b46fd39a8953ef78442bbe0a9a22d5d` added a
direct authority that reparses the raw stderr and independently rederives that
observation before routing and selected-only materialization.

The first complete fake transcript exposed a generic physical-routing defect:
the E0425 composite score map `42/38/18/14` clipped its two leaders to `36/36`
and stopped. Commit `e5c60177f0ae9eaa3765d25b50bf28ab34f60289`
retains the sum of the two individually bounded axes as the raw 0–72 score and
subtracts one shared overflow before the bounded race. Thus
`42/38/18/14` becomes `36/32/12/8`, preserving the winner, top ties, and
non-floored gaps; routes already at or below 36 remain unchanged.

Fresh C9 preregistration `01cd7ce2432d0490098a31b603b08fc3a389437c`
froze spec
`fe726852dc632bba8b64aa6e85bf7ead8d4033bdc82d2b91de148b27f86c0bfc`,
the exact ten-file mechanism closure, 75 prior exclusions, six no-terminal-LF
original/materialized roots, exact rustc parent messages and token inventories,
and the H00→H11→H01→H10 TRAIN grid. Trusted-host preflight authenticated rustc
identity
`aa7c4003101dc045a3689cf9f5c2c91bd0adedf19c9480f8282ce0c9b28efc84`
before either evidence path existed. The exact cohort was then reserved and run
once.

That sole spend terminated
`PASS_C9_RUSTC_DIAGNOSTIC_LEXICAL_CUE_LEARNING_V1` at result root
`901b8d87ed69021d0d7bcf74074310351c0a860aa89453a0cc696018785796db`,
document root
`93d24d1dd270344d94ae5a719bed9663a1ce3f7e0757a445a477f611262d438b`,
and reservation root
`7b77717a38b30b0bf39e91b6903328caddf790dbeec2946f47fd54d934449820`.
Raw result/reservation file roots were
`9a254343979e1bd11472e16d4ff907c954b787a379552651a2fcbfebea520747`
and `dcbe86641dfc1ddfdc0d5f0eecdb08a73a5a6a041e92619b2d8a8d74c669682b`;
evidence commit `8e44dbfba72f44d68eaf809d40a7d8a6958ad337` retains both.

The parent stopped. After two diagonal lessons, both fresh evaluations normally
selected the wrong same-class target; after three lessons, H01 was correct while the
missing H10 still selected H11. These incomplete-grid decisions were retained
as negative telemetry and never executed. The final checkpoint normally chose
H01 for a fresh E0425 identifier at raw `42/38/18/14` and H10 for fresh E0308
at `18/14/10/6`, ordered as correct / same-action-wrong-target /
wrong-action-same-target / both-wrong. For E0425, TRAIN∩EVAL contained exactly
the six stable template words; the fresh identifier contributed `u=0`. Both
selected sources executed before one combined oracle open and matched.
Outcome-off, message-only, and region-only controls stopped; evaluation did not
update the checkpoint. Authenticated work was six diagnostics, six selected
materializations/releases/compiles/runs, four TRAIN feedback openings, one
evaluation oracle opening, four memory advances, no evaluation advance, and no
alternative.

C9 removes the explicit rustc error-code cue and demonstrates
consequence-conditioned binding reuse from stable parent-message word-unigram
roots. Because all four H00/H11/H01/H10 consequences were trained, it is not
held-out composition. It does not learn tokenization, word meaning, the
source-region abstraction, operations, targets, edit semantics, localization,
diagnosis, synthesis, debugging, transfer, SWE, or intelligence. The next
staged question is whether bounded generic source-context events around the
authenticated primary span can replace the supplied FIRST/SECOND root and learn
which still-supplied target marker they bind to under counterbalanced
consequences. Learned source parsing and target-marker discovery remain later.

Commits `54939b2`, `2c7d350`, and
`4ba0cf5caedd48b02bf5afdcb1fe0962cb439270` next introduced a generic
authenticated ASCII-word unigram primitive, a rustc adapter that projected
parent-message and complete primary-span-containing-line tokens into one
role-neutral unordered cue set, and a direct selected-edit authority. Fresh C10
preregistration `55bee8bf7b0e5c3544fcf3910e9abf3e60659a8c` froze spec
`6ebf21e45798066f2a0e6f57d12571cdf15b5b12cba859b7e4fd28c7febb2c97`
for one root-distinct six-source cohort.

Trusted-host preflight authenticated rustc identity
`aa7c4003101dc045a3689cf9f5c2c91bd0adedf19c9480f8282ce0c9b28efc84`
before the exclusive reservation. The sole spend then permanently terminated
`ABORT_C10_RUSTC_DIAGNOSTIC_SOURCE_CONTEXT_CUE_LEARNING_V1`. Evidence commit
`c091872` retains reservation logical root
`772756e588e26d844a83c07cbde9deb635a8f23565982bc18d07bf834874c837`
and raw reservation-file SHA-256
`728abd55430ef4c641b0b48c4adb380a33b3fe7259c4ca3164f89b8c499fc1fc`,
along with terminal-document logical root
`2b1a8d39b7ae8d25f6c680a0d2c9e18ce7e3d9dc4140eea3a87734e1450d9f49`
and raw result-file SHA-256
`481a15d83ee9a5e03909a0c1125089f84c1b321f73c4a98aa025d8cc58d0dde2`.

The runner compiled all six original diagnostic sources before authenticating
the retained diagnostics. The real E0308 parent carried rustc's `function
defined here` child note, but the cohort had preregistered the stronger
`children == []` condition. Authentication therefore stopped before any cue
envelope was admitted and before any learning update, materialization, selected
execution, or oracle opening. The cohort is spent and was not retried.

This terminal artifact localizes an overstrict diagnostic-record assumption; it
does not test whether the role-neutral message-plus-line cues can learn target
bindings. C10 supplies no cognitive observation, learned association, selected
repair, consequence, transfer, SWE, or intelligence evidence. A successor must
be a separately preregistered, root-distinct C10b whose parent-only adapter
deliberately ignores child diagnostics. C10 itself remains immutable.

The separately preregistered C10b successor
`40c0287596973b093b534966fb6343fa711c72bd` kept mechanism commits
`54939b2`, `2c7d350`, and
`4ba0cf5caedd48b02bf5afdcb1fe0962cb439270`, but used fresh root-distinct
sources and a parent-only diagnostic contract. Cohort source and test SHA-256
values were
`767bbe12f9a7470ff44a291e078831b99bf695d26f259bb528350d83c3367ca0`
and `b313e49c36d32f4c507c18ca54a393bd61faceaccd184efaec67faab98af395c`.
Spec `2e74503179f8c237837cf030091a14b050b8e9c40061277f8c3f263e18a6f2ed`
froze 99 excluded prior roots and new opaque contexts `tirvax` and `molqen`.

The sole trusted-host spend authenticated rustc identity
`aa7c4003101dc045a3689cf9f5c2c91bd0adedf19c9480f8282ce0c9b28efc84`
and terminated `PASS_C10B_RUSTC_DIAGNOSTIC_SOURCE_CONTEXT_CUE_LEARNING_V1`.
Evidence commit `b8f93d0ee5d159dc70366cfe2ae916708b050ed4` retains reservation
logical root
`8af55f22aaf825eba6a7d76ccfee126769c36b13243b5b009323d5a1f0e34c3c`
and raw reservation-file SHA-256
`1f0e89cbb3f39d6cafdfc0c6622b3a389b6587e9b3907f1d0596df09d79f7324`,
result logical root
`1148701f113011ac82d5c0362138032b359a2d21550a753c5a3921a9cf34034e`,
document logical root
`6f071866702ac4563acd1073f8a6caebf294cced1a5b2c187c23dfa4fe5cb8c5`,
raw result-file SHA-256
`6fa32880fdbf1406ac868f2881d32adb373adc1aecfd8be5efe572cda044eee4`,
and work root
`de3e1501929def717a288d22006a7b2b19ff1824f473e0d728c35233a38571d4`.

Four explicit TRAIN consequences covered the complete H00/H11/H01/H10 grid.
The 25 distinct TRAIN cues produced exactly 100 cue×axis×component binding
rows, and all seven evaluation-only cues retained zero support. Exact supported
evaluation intersections were `(cannot, find, in, map, molqen, scope, this,
value)` for H01 and `(map, mismatched, tirvax, types)` for H10. The full
checkpoint normally selected H01 and H10. Ordered as correct /
same-operation-wrong-target / wrong-operation-same-target / both-wrong, their
raw score maps were `50/46/26/22` and `26/22/18/14`; the shared-overflow
physical race mapped H01 to `36/32/12/8` and left H10 unchanged. All six
selected materialized sources compiled, ran, and matched their expected
outputs.

Outcome-off, message-only, and source-line-only controls stopped. Reordering
the cue input produced the same canonical unordered observation, context
ablation stopped on insufficient target evidence, and context derangement
routed H01/H10 to H00/H11 without executing those wrong targets. Evaluation
did not advance the checkpoint. Authenticated work was six diagnostic compiles,
six selected materializations/releases/compiles/runs, four TRAIN advances, two
evaluation executions completed before one shared oracle opening, zero
evaluation updates, and zero alternative executions.

C10b establishes consequence-conditioned binding reuse from authenticated
parent-message and complete primary-span-containing-line generic word unigrams,
removing C9's supplied relative-region cue. It is not held-out composition
because every hypothesis was trained. Target markers and occurrence bindings,
primary spans, tokenization, the concrete operation bank, exploration choices,
edit semantics, and outputs remained host-supplied. The result is not language
or Rust semantics, localization, diagnosis, synthesis, debugging, transfer,
SWE, intelligence, or external-model evidence. C11 should next learn target
occurrence/localization rather than receive marker bindings.

The first preregistered C11 occurrence-localization spend
`30abfa336cd5451c2fba458b9b318b56c8cb6e3b`, exact spec
`4070de660060756bdf20910e8e2dee2d857e9249c1fae0e3522ecfe5e64f02bb`,
terminated before its first diagnostic subprocess because the production
wrapper referenced nonexistent `rust_executor.COMPILE_TIMEOUT_SECONDS`.
No diagnostic compile, cognition, materialization, selected execution,
feedback, or oracle opening occurred. Evidence commit
`876d2b87fb97a527557d4e740569c2a82acd284c` retains the terminal abort; the
cohort remains spent and immutable.

The separately preregistered, source-root-distinct C11b successor
`51bafcefd0a6a81b5d5a49fd53b39a967b73b312`, exact spec
`c2ab03e2d9e8824d490602219f22c6e5a01d4ef402f7d589bb89e144a9dc9636`,
used a comprehensive 143-root exclusion: 111 inherited roots plus the stale
C11 fake transcript's six originals and ten materializations, and committed
C11's six originals and ten possible materializations. It terminated PASS at
result root
`de571775a22e39eeb7f89b73f4cec81e19b910fac24a1e2802ccb9fe1646cf11`;
evidence commit `638407c110ba456083e1122e7bc91b2867cf5656` retains the result.

Across four paired outcome-blind TRAIN lessons, both FAR/NEAR candidates came
from the same pre-lesson checkpoint and executed before one paired feedback
opening; exactly the passing outcome advanced memory, while the failing outcome
was retained but never advanced. The learned checkpoint then selected two fresh
occurrence-addressed edits, both of which executed before one shared evaluation
oracle, with no evaluation update.

C11b establishes one narrow case of consequence-conditioned selection of an
authenticated relative preceding-`let` occurrence from parent-message and
primary-line word cues inside a supplied 2x2 operation-relation bank. All four
hypotheses were trained, so this is not held-out composition. Tokenization, the
simple-`let` grammar, primary spans, FARTHER/NEARER ontology, exact-two policy,
operation interventions, exploration, edit semantics, compiler/executor, and
output oracle remained supplied. This is not learned Rust semantics, generic
localization or diagnosis, synthesis, debugging, transfer, SWE, or intelligence.

In the scale lane, commit `018b3a3` reduced full-touch GID scratch from `R` to
`min(R,Q)` by streaming authoritative resident/spike arrays. It saves
`8*(R-Q)` bytes (523,776 in the representative 65,536/64 case) without changing
event-proportional behavior. The arithmetic is useful, but 72R resident state
and the packed ledger still make physical 100-billion residency infeasible; the
next useful step must be structural rather than another scratch-only claim.

Commit `9aa4d10` made resident and interval-ledger physical backing grow with
occupancy rather than configured logical capacity. Empty 65,536- and 100-billion-
capacity runtimes both occupy 10,040 bytes; the same four high-GID cells occupy
10,384 bytes with exact dynamics and overflow parity. It removes 5,267,456 eager
bytes at 65,536 and 80,375,000 at one million, but a fully materialized 100-
billion runtime still computes to 8,037,500,000,000 bytes. The result is sparse
logical construction, not physical 100-billion feasibility.

Commit `d90ebbe` removes the next construction barrier by making both
event-proportional phase buffers grow with peak stepped occupancy. Empty payload
falls from 1,058,104 to 9,528 bytes at `R=65,536`, an exact `16R` saving; the
same focused high-GID traffic at logical 65,536 and 100 billion produces exact
eager/lazy parity and the same 16,200-byte payload with 64-slot buffers. The
nominal construction saving is 16,000,000 bytes at one million and 1.6 TB at
logical 100 billion. This is focused construction/payload evidence only, not
throughput, capability, or fully materialized 100-billion residency.

Two immediate throughput hypotheses then failed their predeclared gates and
were fully reverted. A resident GID/dynamics hot-cold split measured -1.49% at
131,072 residents and +0.55% at 1,048,576. A target-sorted adaptive resident
cursor measured +7.1% for 1,024 events but only +0.62% at `K=R/16`; execution
stopped before the larger/poison branches, so they carry no claim. The exact
`d90ebbe` source and hash were restored after each STOP. These results redirect
the speed lane toward materially batched event/state work rather than another
scalar storage or lookup substitution.

The next coherent batch prototype grouped canonical due-target runs and replaced
the dormant B-tree with a packed sorted base plus sparse mutation overlay. Exact
state, queue, spike, non-payload counter, binary64 cancellation order, and first-
overflow partial-state parity all passed. Dense delivery improved 1.242x and
H=32,768 dormant backing fell from an accounted 1,835,008 to 786,432 bytes, but
dormant reactivation reached only 1.183x against a 1.20x gate, the four-step
N=8,192/F=4 recurrent trace reached only 1.026x against 1.10x, and idle work
regressed 10.1%. The entire prototype was reverted to source hash
`1651a4325bb271329e421501ccfb6a934b4a9f507ae8cf10ec3ebde160ec9906`.
This closes standalone target/dormant batching; source/connectivity generation
or a combined state-generation kernel is the next speed hypothesis.

A four-lane Philox source-generation successor preserved exact events, weights,
retry/failure order, recurrent state, and payload while improving its isolated
generation phase 1.278x. It nevertheless regressed the required four-step
generate-and-deliver trace to 0.907x (K=64 no-spike was 0.996x), so it too was
fully reverted to source hash
`1651a4325bb271329e421501ccfb6a934b4a9f507ae8cf10ec3ebde160ec9906`.
This closes standalone source ILP and motivates one fused
generate/canonicalize/deliver experiment rather than another phase-local patch.

The fused attempt preserved exact parity and payload while reaching 2.443x
canonical ordering, 2.478x equal-target delivery, and 1.356x on the four-step
trace. It still missed its generation-plus-canonicalization gate at 1.021x and
regressed idle work 3.3%, so it was reverted. A second attempt kept scalar
source generation and isolated only radix ordering plus target-run delivery;
the dense gains improved to 3.577x, 2.515x, and 1.296x, but K=64/no-spike and
idle regressed 28.3% and 9.2%. It too was reverted to the same source hash.
The dense kernels are credible; the remaining hypothesis is hybrid dispatch
that preserves the original scalar code path for small and idle workloads.

Commit `56d22a5` passes that hypothesis with one fixed 4,096-event threshold and
no sweep. Non-inlined stable target radix and equal-target delivery helpers
retain the existing scalar source and small/idle paths. Exact focused parity and
unchanged 5,181,240-byte payload accompany 3.199x ordering, 2.289x delivery,
and 1.297x four-step recurrent gains; K=64/no-spike is 1.005x and idle 0.949x.
The result is retained CPU FULL_TOUCH acceleration, not broad runtime or
physical 100-billion-cell evidence.

A bounded successor tried to reclaim the separate submitted-event scratch by
phase-aliasing submission canonicalization, radix work, staged cells, and merge
through `event_work`. Focused parity and the exact 1,310,744-byte payload saving
passed, but retained absolute timing did not: recurrent 4,366,300 ns, K=64
1,488,500 ns, and idle 4,258,000 ns all exceeded their fixed baselines. The
prototype was fully reverted to procedural source SHA-256
`33dbc45dd1fcc7584ca4b6650be152c3cbcc61e6c9ef4465867b751b15e6a68f`.
This negative result rules out that alias as a standalone successor; it does not
rule out a materially broader fused kernel.

Commit `e096d2d` passes the bounded broader hypothesis. A private advance-local
gate requires FULL_TOUCH, delay one, a following step in the same call, no live
future prefix, at least 4,096 generated events, complete identity-ordered
residency, and successful queue/generated/delivered arithmetic preflight. The
unchanged source-major generator still writes authoritative pending first; only
complete success applies that tail directly to target residents in the same
per-target source/slot order and clears it. Exact raw resident, pending, spike,
dormant, counter, poison, and final-boundary parity passed with one fused
handoff and the unchanged 5,181,240-byte payload. The nine sorted retained→fused
release samples in ns were recurrent
`[3249400,3512200,3543100,3671600,3735200,3804700,3878400,3987400,4230000]`→
`[2496100,2556200,2628600,2749500,2829500,3001700,3222800,3223600,4663400]`,
K=64 `[809700,845600,891700,974300,1089100,1132200,1317800,1348300,1544800]`→
`[788700,788900,802100,1005500,1016000,1297700,1443700,1462200,1498500]`,
and idle `[1636700,1934600,2302000,2370700,2379600,2625000,2739000,2780200,2999200]`→
`[1569000,1581300,1748700,1887600,2128800,2327200,2371500,2446900,3051800]`.
Medians were 3,735,200→2,829,500 ns (1.320x; ≥1.15x and ≤4,313,022 ns),
1,089,100→1,016,000 ns (0.933x; ≤1.03x and ≤1,394,929 ns), and
2,379,600→2,128,800 ns (0.895x; ≤1.03x and ≤2,797,480 ns). Source SHA-256 is
`e93cd287bd8dfcda4c4c9b1db45fcdb4746c4113daec9d7da89f799a7cacfb99`.
This is one focused CPU recurrent-handoff result; all excluded shapes retain
the prior path, and no broad runtime or physical 100-billion-cell claim follows.

The immediate lazy-fallback-seed successor deferred its Philox seed until 128
ordinary target collisions. Exact parity, forced collision/failure, and counts
passed (dense recurrent eager 8,192 versus lazy zero; forced lazy fallback one),
but its only release run failed: recurrent 3,048,900→3,055,000 ns (0.998x;
required ≥1.05x and ≤2,829,500), K=64 1,567,400→1,556,700 ns (0.993x; absolute
≤1,046,480 failed), and idle 3,045,300→3,246,800 ns (1.066x; required ≤1.03x
and ≤2,192,664). The tranche was fully reverted to `procedural.rs` SHA-256
`E93CD287BD8DFCDA4C4C9B1DB45FCDB4746C4113DAEC9D7DA89F799A7CACFB99`.
This is a narrow rejection of lazy seed initialization alone, not of broader
generation changes.

## 11. Common G0 substrate begins

Commit `4843236` adds a domain-neutral opaque-byte projection with 65,536 input
cells, sparse value/position/binding assemblies, recurrent trace lanes, and one
shared factorized fitted-GLIF output race. Commit `9ad36b3` adds a frozen local
one-byte associative memory over ordered event traces. Regulatory- and
spatial-shaped quarantined fixtures recall through one checkpoint; reversed or
unknown inputs STOP, teacher/cell-off commit no ticks, and contradictory teachers
produce a physical collision. This is projection and direct association only:
factorized relation composition, adjudicative cohorts, the nine-artifact
binding, evaluator, and G0 result remain absent.

References:

- [`common_event_cells_v1.py`](../../experiments/g0/common_event_cells_v1.py)
- [`common_event_associative_memory_v1.py`](../../experiments/g0/common_event_associative_memory_v1.py)

## 12. Target-teacher-free integrated coding sessions

The first persistent R4 toy coding session replaced the earlier answer-bearing
bridge design before spending any target task. Its learner and no-update control
were independently grown from the same eight-case public corpus frozen before
the target cohort existed. The environment exposed exact starter source,
natural-language task text, one generic whole-source tool, and only coarse real
sandbox consequences. It supplied no target repair, candidate patch, placeholder
rewrite, private stage policy, DEV, or protected material. A pre-run reservation
committed code, corpus, stream, protocol, worker, and Python-runtime identities.

The sole spend stopped at a clear null result: 0/10 tasks, 20/20 treatment
attempts and 20/20 frozen-control attempts at `signature_error`, no successful
receipt, no learning event, no digest change, and no second-attempt recovery.
Post-run action-hash decoding against the frozen prior mapped every treatment
decision to the same public `fallback_lookup` source. The current byte/phrase
cell model therefore retrieves one whole action but has no computation that
binds novel observed symbols into generated declarations/body references or
uses a failed receipt to revise that action.

The spent stream root is
`9007ca07de5bf09195741ca3a586512c4aa35f4aafde74395c7a9660897db741`;
the result artifact is
[`r4_integrated_continual_coding_session_v1.json`](../../benchmarks/r4_integrated_continual_coding_session_v1.json)
with SHA-256
`71e0825252bb2cb51fbb473993204195a810acdc3617315cc259bb12fbe83a9d`.
This cohort is immutable and must not be rerun. The localized successor is one
fresh causal alpha-binding gate based on neuron-emitted input pointers, followed
immediately by a fresh integrated cohort; entrypoint macros and target-family
source banks are not licensed.

That source-local micro-gate was subsequently spent once. Its semantic-role
tuple was removed before inference: both growth and fresh evaluation derived
ordered dynamic vocabularies mechanically from the exact starter source. The
frozen gate failed its strict exact-source criterion at 4/8, while every neural
output contained pointers, the raw binding-off control retained 0/8 exact
bindings, and the empty-inventory control stopped. Offline analysis of only the
frozen artifact found correct entrypoint/parameter binding on 7/8 and exact
causal rotation for all seven addressable pointer programs. One wrong behavior
selection emitted indices outside a shorter inventory; three others emitted a
valid but semantically wrong slice body. The behavior confusion map was
`0->4, 1->6, 2->6, 3->3, 4->4, 5->6, 6->6, 7->7`.

The result therefore banks a partially useful general copy mechanism and
localizes the remaining failure to alpha-sensitive prompt/template retrieval,
not source materialization. The immutable artifact is
[`r4_source_local_alpha_binding_gate_v1.json`](../../benchmarks/r4_source_local_alpha_binding_gate_v1.json),
SHA-256
`f08a7fa57455664c9525833ce310e7b1f8166cdb5b5ea833e10d4e46d4d57de2`.
The roadmap now returns to integrated pressure: bidirectionally normalize both
observed prompt and emitted action through the same derived pointer vocabulary,
then use a separately named fresh execution cohort. The spent alpha stream is
not a tuning surface.

That separately named V2 integrated cohort was then spent once and frozen. The
treatment and independently grown no-update control each began from all eight
public-prior cases plus one generic STOP example and received the same twelve
fresh public-TRAIN tasks. Both arms finished 0/12 after 21 attempts apiece. The
treatment distribution was eight `regression_failure`, eight `runtime_error`,
two `signature_error`, and three fail-closed
`pointer_binding_failed_closed` stops; control was identical. Neither arm
learned, changed digest, or recovered on a second attempt.

Relative to V1's twenty `signature_error` outcomes, source-local binding
materially moved sixteen V2 attempts into real regression or runtime execution.
The semantic policy did not improve: every raw action still selected the frozen
public-prior `fallback_lookup` template, and each available second attempt was
unchanged. The result therefore localizes the next frontier to semantic
episode/template retrieval plus failure-conditioned revision, rather than more
copy binding or replay of this stream.

The V2 stream SHA-256 is
`99e11062d7d4ede454abd0c5b52747ca208ea9a25a917088c2c9e14c574e5253`.
The immutable artifact is
[`r4_integrated_continual_coding_session_v2.json`](../../benchmarks/r4_integrated_continual_coding_session_v2.json),
SHA-256
`5e85b7f81007366fca2c0557814858a7bbe2e4cea5d5f70028fec3e7fc25542c`.
It must not be rerun or tuned. This is a public toy failure-localization result,
not SWE evidence or a mesoscopic integrated forcing-gate result.

V2's semantic-retrieval failure drove one reusable successor mechanism rather
than another task-family patch. V3 assigned each public prior episode an
independent pointer-bound cell route, converted only mechanically tokenized
public-task text into route currents, selected through a physical dendritic
race with a tonic STOP competitor, and conditioned generation on the winning
episode. Failed real receipts applied transient one-route refractory exclusion;
successful receipts could update the selected episode. The source/corpus freeze
records `target_tasks_present_at_freeze=false`, eight public-prior action routes,
one generic STOP route, and no duplicate route repetition.

The fresh V3 stream contained twelve independently authored public-TRAIN toy
tasks: four prior-capability transfers, four recombinations, and four unsupported
algorithms. Treatment and independently grown frozen control both solved only
the same inner-slice transfer: 1/12 overall, 1/4 prior transfer, 0/4
recombination, and 0/4 unsupported. Treatment used 18 attempts with one pass,
seven `sandbox:regression_failure`, one `sandbox:runtime_error`, one
`sandbox:signature_error`, one `sandbox:policy_violation`, and seven
`stop:pointer_binding_failed_closed`. Control used 16 attempts with one
`passed`, four `sandbox:regression_failure`, two `sandbox:runtime_error`, two
`sandbox:policy_violation`, and seven `stop:pointer_binding_failed_closed`.

The treatment did change internal state and action selection. Its one successful
receipt produced one learning event and changed the model digest from
`a498d0dee49e8c5a61c543bfbc439470e75fa81dae5c4894679de82498374d7f`
to
`06e3bb9498c826ed5085110bc47bc91105d921f8fd2ebc8de7ac3433cf4cfc53`;
the control digest stayed at the initial root. Ten failure events produced ten
refractory attempts, and six treatment second attempts changed versus zero in
control. None recovered. Treatment selected seven distinct action routes,
control selected all eight, and neither arm selected the explicit STOP route.
Thus V3 demonstrates narrow physical cue routing and failure-conditioned action
change, but not continual capability improvement, compositional reuse,
successful revision, or learned uncertainty.

The V3 stream SHA-256 is
`eb4d92dc9e9c0a1c1082cfe522d246e787de5917c12979a2f7a3eb50158d4d64`;
the bootstrap-corpus SHA-256 is
`8f78ef7d75a00309e3c7c74cdb3c681e5f9e7b9130064dfd04fcd462148f54bc`.
The sole artifact terminated `COMPLETED` at
[`r4_integrated_continual_coding_session_v3.json`](../../benchmarks/r4_integrated_continual_coding_session_v3.json),
SHA-256
`effb9053111826481645cea593d7ca1bc840bed2085a3258744563e9aa633c9d`.
Its recorded claim boundary is `prepared fresh public-TRAIN routed continual toy
pressure only; no protected evaluation or mesoscopic forcing-gate claim`. The
cohort is immutable and must not be rerun or tuned. The next integrated
successor should test a general composition/abstraction mechanism, contrastive
uncertainty/STOP learning, and revision that can recover after execution
failure, on another fresh cohort.

### Shared recurrent factorized output binding

The V3 result localized its main architectural error before another integrated
cohort was opened: eight independent episode anchors could choose and revise
whole prior actions, but could not construct a new action from reusable pieces.
The next mechanism therefore replaced whole-output WTA with one shared recurrent
fragment graph. `src/humanbrain/factorized_pointer.py` learns generic lexical
fragments, a common-subsequence backbone, variable optional spans, and signed
cue evidence on recurrent `(canonical context, target cell)` edges. Actual cells
carry progression within an optional span; only its terminal edge rejoins the
canonical context. Query inventory symbols and repeated output-local roles are
materialized after their fragment cells win, with one physically raced fresh
symbol reused for every occurrence of a learned local role. A recurrent STOP
edge remains independent of cue drive, and exact training episodes survive only
as provenance hashes.

The separately frozen opaque fixture contained six public lessons and three
fresh cue-factor combinations. Its exactly-once gate spent an 82-cell,
422-branch model and returned
`PASS_FACTORIZED_OUTPUT_BINDING_GATE_V1`: treatment and fresh-local binding were
3/3; whole-episode-only stopped 3/3; binding-off emitted 3/3 exact placeholder
forms but 0/3 bound answers; fragment-projection-off stopped 3/3; and every one
of the 15 required fragment/STOP decisions carried a physical spike trace. An
unknown cue stopped with STOP present and emitted a construction under the
STOP-off control. Initial and final model digests were identical, so inference
was read-only.

The immutable artifact is
[`r4_factorized_output_binding_gate_v1.json`](../../benchmarks/r4_factorized_output_binding_gate_v1.json),
SHA-256
`301e40786f47a9e6ce09d03e7049524faa2bed9d6d928c66dd57cb2e7ce33376`.
The frozen model, gate, and fixture source SHA-256 values are
`595b4ab5b8bb486e843f52fbd73e4958d5e3d6d0efad6cb690491757a94d1017`,
`4d37da1ff9f82915438b22f579c106cf7d12f3e15df3d77e3234237cdb8e930a`,
and
`1a34614ccb94b336c24a5fc0c8a5ab1c3d0d0f2817ed2c59de977e7e796ac5da`;
the fixture content root is
`654545391474f40b8b4b6e41a2485e5f86b0a18475fd629a343773badbf036e6`.

Before integration, the same decoder was generalized without task-family or
Python-AST labels to accept all eight variable-length public-prior tool outputs.
Developmental self-replay was lossless on 8/8: seven byte-exact and one modulo
a consistent bijective fresh-local rename. A bounded profile then found that
serializing the immutable 18,686-branch digest consumed 0.260 of 0.319 seconds
for eight decodes. Caching that derived digest with plasticity invalidation
reduced the same measurement from 266.635 ms to 60.045 ms, a 4.44x improvement;
the remaining 529 physical races consumed 0.049 of 0.059 seconds.

This result banks arbitrary-token two-factor construction, input projection,
fresh-local reuse, and causal STOP at microfixture scale only. The gate executed
no generated Python or sandbox, saw no V4 task, persistent learner, target
teacher, private, or protected input, and says nothing about SWE or continual
coding capability. Its online feedback can revise existing selected cue edges
but cannot grow a previously unseen structural fragment. The gate and fixture
are spent; the next question belongs to a fresh persistent integrated cohort.

### Fresh factorized continual V4 session

That fresh cohort was then frozen and spent exactly once. One persistent
factorized treatment learner and one independently grown no-update control began
from the same eight public-prior successes. Both received the same twelve
public-TRAIN tasks with one uniform three-attempt ceiling: three renamed
transfers, three factor recombinations, three feedback-revision tasks, and
three insufficient-evidence requests. Frames contained one canonical
typed-signature block and a source-derived inventory; only exact sealed learner
programs entered the sandbox. Terminal policy/tests opened only after an action
was sealed, while stage, group, authored budget, and feedback labels opened
only after both arms had completed every decision.

Both arms scored 5/12. Each solved 2/3 renamed transfers, 0/3 factor
recombinations, 0/3 feedback revisions, and 3/3 insufficient-evidence requests
with immediate explicit STOP. Treatment used fourteen attempts—nine explicit
STOPs, two passes, and three regression failures. Control used sixteen—eight
explicit STOPs, two passes, three regression failures, and three signature
errors. Neither arm recovered after a failure or gained a late uncertainty
success. Treatment therefore had zero capability advantage.

The treatment did apply five authenticated updates: two positive updates after
the renamed passes and three negative updates after scaled-distance regression
failures. Its digest changed and branch count grew from 18,686 to 18,874; the
control digest and 18,686 branches were unchanged. However, the three negative
updates emitted the same unscaled distance program on all three attempts.
Artifact fields marked one treatment and two control tasks `retry_changed`, but
the choice digest also commits the changed frame/model identity. Direct frozen
output comparison corrects the interpretation: raw and materialized output
changed on 0/2 treatment and 0/4 control adjacent retry comparisons.

The same failures harmed retention. Before V4, treatment replayed the public
prior at 7/8 byte-exact, 8/8 alpha-equivalent, 8/8 valid tool outputs, and 8/8
terminal STOPs. Immediately after scaled-distance task index 3, prior case 1
even-filter replay changed to explicit STOP and those counts fell to 6/8, 7/8,
7/8, and 7/8 for the remainder of the run. Control stayed at 7/8, 8/8, 8/8,
and 8/8. The
untrained unknown-cue probe retained explicit STOP in both arms at all thirteen
checkpoints.

Physical traces sharpen the structural failure. Every one of the 30 attempts
contains a STOP winner: 13 terminate a complete tool output and 17 select early
explicit STOP with at least one competing continuation. Eight early traces
with two-symbol inventories stop at fragment step 21 after one parameter and a
dangling comma; eight with three-symbol inventories stop at step 24 after two
parameters and a dangling comma. The remaining treatment trace stops at step
30 on the five-symbol capped-affine task after closing a three-parameter
signature and reaching its body indent. Successful renamed outputs likewise
use three parameters, while the control capped-affine output omits the required
fourth parameter and fails its signature. The decoder has composed lexical
pieces but retained a fixed-three signature pattern rather than learning an
arity-sensitive relation.

The six supported early-STOP cases first localized a missing
variable-cardinality traversal relation. The repeated scaled-distance failure
and retention loss separately falsify existing cue-edge plasticity as a
sufficient continual-learning rule: authenticated consequences still need to
form or revise reusable structural relations with local credit. The first
localization received the developmental mechanism gate below; neither result
permits retuning or rerunning V4.

The immutable artifact is
[`r4_integrated_continual_coding_session_v4.json`](../../benchmarks/r4_integrated_continual_coding_session_v4.json),
SHA-256
`15af435b757a9e0aceab024a88e5ff8bfe013e51020c514d77b0edb1138874f7`.
Runner, stream, stream-custody, and task-source SHA-256 values are
`112435a42f6c85e3eddb057ae86983df6578ddd2bd90f736ae76e5ae26678501`,
`b0e77bb4f29c6eff7eb717f9fda9d77f72a68f9954294dbdfd4be1dc0d962d57`,
`b54ad8d7efa7d28b52951e03734d1cb7fdc9c0f191e8dd41fa0314c935c77b3b`,
and
`520ea0ad97f5f211295ccd48b7900f4eaa475bcb73bd8d7a5816cba5165e552f`.
Reservation and final corpus/execution custody match exactly and pin the
factorized model at
`595b4ab5b8bb486e843f52fbd73e4958d5e3d6d0efad6cb690491757a94d1017`,
protocol at
`f7bc723457bdb733bb560feb7e7f509b39cb756b2eec130e7ead6168e0bcb2e2`,
worker at
`258aaf45decb8ed6635a85b92ee970591932b2cf53b691149a0d66777601c870`,
and Python 3.11.15 executable at
`6765c6b1685c86877fabe14d82240f3fab2913a617a85e8d09e61dc40798013b`.
Its claim is limited to fresh developmental public-TRAIN integrated
failure-localization: no protected evaluation, family-disjoint transfer,
Simple-SWE M0, or completed mesoscopic forcing-gate evidence.

### Developmental recurrent variable-cardinality inventory loop

V4's fixed-three signature traces motivated one smaller arbitrary-token
mechanism. `src/humanbrain/factorized_pointer.py` now discovers a repeated
inventory relation from demonstrations and represents it with one recurrent
item cell rather than one position cell per demonstrated item. Learned
separator transitions feed back into that cell. After each item, physically
raced continue and exit cells combine learned occupied-next-slot and exhausted
weights; positive learned cue-familiarity drive conditions the initial non-STOP
race and loop entry. Pointer binding remains a separate post-selection
operation.

The model learned from eight public demonstrations at cardinality three and
grew to 80 cells and 90 branches. On scoring cardinalities 1, 2, 4, and 5,
treatment was exact 4/4. The fixed-arity loop-off control was exact 0/4 and had
no loop races. Binding-off emitted the expected placeholder structure 4/4 but
the bound target 0/4. Unknown-cue and empty-inventory probes stopped 2/2, and
all 14 complete scored traces had physical custody. The treatment and
binding-off loop contracts contained 24 physical continue/exit races, and the
two probe STOPs were physical. The initial and final model digest both remained
`7359928420927590c144792fdd86df8c98bf60b09bd6502d9ec3d4f219d9d086`,
so scoring inference did not mutate learned state.

This is **not fresh held-out evidence**. The artifact explicitly records
`cohort_novel_at_artifact_spend=false`: the focused test
`tests/test_factorized_pointer_inventory_loop.py` exercised the scoring cases
before artifact spend. The result is therefore a public developmental
mechanism check only, not evidence for integrated V5, continual improvement,
SWE, protected transfer, or broad generalization.

The spent immutable artifact is
[`r4_variable_cardinality_inventory_gate_v1.json`](../../benchmarks/r4_variable_cardinality_inventory_gate_v1.json),
SHA-256
`99da8382989e2fe8008f05fac06de4cdcf415269ad2fa462b311a23a15fc211a`.
Fixture-content, fixture-source, runner-source, and factorized-model-source
SHA-256 values are respectively
`656424b1e47347636c0b6638ba9b799b9aac8db5a3fef8df489645818acc67e7`,
`6b73f64371ce957328a0fb5d6cd08decc212ad1b6f04a99aa0a8e9bf24565fee`,
`edb7144142c68b59aa2608435abc2ece681a88855e8945e832c42abb5b9a6950`,
and
`d34ed27d64600c0ee8e903425aebd4568fbb0a31beab848f8badc9cb32cbde36`.
The next evidentiary step is a genuinely fresh integrated V5 cohort, not a
rerun of this developmental fixture. That cohort should establish whether the
loop removes V4's signature-cardinality failures and use its new failure
distribution to choose the next reusable mechanism.

### Fresh recurrent-inventory continual V5 session

That cohort was authored before any V5 model output or runner existed, frozen,
independently audited, and spent exactly once. The stream contained sixteen
public-TRAIN tasks: four renamed transfers, four factor recombinations, four
execution-feedback revisions, two delayed-retention rechecks, and two requests
whose evidence was intentionally insufficient. Arity 1, 2, 4, and 5 each
appeared four times. Public frames contained commitment, path, signature,
prompt, and inert starter source only; target answers, successful target
sources, candidate patches, answer-bearing diagnostics, private authored
budgets, DEV, and protected inputs never entered learner cognition.

One persistent treatment learner, one independently grown no-update control,
and one independently grown inventory-loop-off arm began from identical
122-cell, 18,762-branch, 274-token models with training digest
`76a2c8d29670c14212807a7b0012a7dd83394ab40450d82da00003cfc73ba65e`.
The two main arms used one uniform three-attempt ceiling. Loop-off ran only one
original-frame attempt after both main arms had sealed each task, with no
feedback, update request, or retry.

Treatment/control/loop-off scored 5/16, 6/16, and 2/16. Their stage counts were:

- renamed transfer: 4/4, 4/4, 0/4;
- novel factor recombination: 0/4, 0/4, 0/4;
- execution-feedback revision: 0/4, 0/4, 0/4;
- delayed retention: 0/2, 1/2, 0/2;
- insufficient evidence: 1/2, 1/2, 2/2.

The inventory loop crossed the developmental-to-integrated boundary. On the
fourteen executable tasks, control first attempts had exact entrypoint and
ordered-parameter interfaces and entered the sandbox on 12/14. By arity its
exact-interface counts were 3/3 at one, 4/4 at two, 2/4 at four, and 3/3 at
five. Loop-off was exact 0/14, entered the sandbox 0/14, and selected physical
explicit STOP on all tasks. Both main arms passed renamed transfer 4/4 across
all four arities. The two control misses at arity four belonged to the same
unsupported shared-even family, so the remaining error was not an arity trend.

The semantic failure was repeated and cross-group. Exact-interface outputs
selected one familiar primitive while omitting the required composition:
slicing without normalization, absolute distance without scaling/surcharge,
one-input iteration without shared nonzero product, or distance over the wrong
affine roles. Both arms scored zero on all eight recombination and revision
tasks. This falsifies flat independent cue-to-fragment projection as sufficient
for hierarchical reuse even after variable-cardinality structure is available.

Treatment applied fourteen authenticated digest-changing updates: four success
and ten failure. Cell count and vocabulary stayed fixed, branches grew to
19,270, and the final digest became
`d9ca6b237609095648e9eae4848591b4785ab2dc650fe68f67a5c69b6785caaf`.
Control and loop-off remained identical to initial. Direct retry comparison
found eight treatment adjacent pairs across five tasks: five
`repeated-identical-failure` and three `changed-unsuccessful`; every change was
failed executable code to explicit STOP. No retry produced a different
executable program and no recovery occurred. All sixteen control pairs repeated
the same failed program. V3 changed whole routes without recovery; V4 repeated
one fragment program; V5 can now suppress a fragment construction, but still
cannot assemble and test a constructive alternative.

Retention separated useful success updates from destructive failure updates.
Treatment stayed at 7/8 byte-exact and 8/8 alpha-equivalent/valid/terminal after
all four successful transfers. After scaled-route-cost failure updates it fell
to 6/8 and 7/8, then after the final unsupported-action failure update it fell
to 5/8 and 6/8. Control remained at 7/8 and 8/8. Prior cases 1 and 7 were lost;
delayed task 12 passed in control but treatment stopped. The unknown-cue probe
still stopped at every treatment/control checkpoint, so failure updates damaged
supported constructions without deleting the generic uncertainty response.

All recorded neuronal custody checks passed. The 72 task-attempt traces contain
3,747 fragment races, 172 treatment/control inventory-loop races, and 72 STOP
winners. The 306 retention traces contain 17,485 fragment races, 816 loop races,
and 306 STOP winners. Winner membership, output equality, digest binding, and
read-only retention invariants had zero violations. Reservation and terminal
corpus/execution custody match exactly.

The immutable artifact is
[`r4_integrated_continual_coding_session_v5.json`](../../benchmarks/r4_integrated_continual_coding_session_v5.json),
SHA-256
`264541859cbd901e287ec5f9b88e8eedbb66bb51268b5aafe2a95b65c139ebfc`.
Runner, task-stream source, train-stream, train-stream-custody, and model-source
SHA-256 values are respectively
`f474438bec24e677762c03f135aacf7ded8a02846ed22bb27e99d51070fb2f59`,
`c73b173265f2d4478fc03a9fffbd6d4d657e4f21aa29e0a47fcb89152ca3d8a4`,
`f5abad0bd725b6946ea9ee98e74752c03940bcfe615de8ef1caa0c598b51d1e2`,
`4081b581af922b78c0cffbc6d4241b474b3a65f1c571e998d0a90bdeb08ad495`,
and
`d34ed27d64600c0ee8e903425aebd4568fbb0a31beab848f8badc9cb32cbde36`.

The result is fresh public-TRAIN integrated failure localization, causal
inventory-loop integration, and renamed transfer only. It is not useful
continual improvement, novel composition, constructive revision, stable
retention, a completed mesoscopic forcing gate, protected transfer, SWE, or
external-model evidence. The next mechanism is one bounded recurrent
compositional assembly stack over arbitrary learned fragments with physical
stack decisions, stack-off ablation, and assembly-scoped eligibility. It should
receive one small opaque causal gate before integration. That mechanism gate is
recorded below. Broad route inhibition and provenance consolidation remain
deferred because V5 exposed no viable alternate assembly and no useful
treatment advantage to protect.

### Visible-scoring developmental recurrent assembly-stack gate

The localized mechanism learned from complete opaque successful experiences,
without supplied spans, groups, roles, stack actions, candidates, task labels,
or Python semantics. Four public demonstrations formed a parity subset over
three binary axes. Their seven-token responses placed the three axes in nested
intervals around one equality-derived payload copy. A fifth public experience
supplied a direct `ATOM` payload-copy route. The two scoring cases recombined
factor triples absent from model training and used new payload symbols.

Treatment emitted the exact nested response on 2/2. Both treatment traces
reached physical depth at least two with PUSH and POP winners. `STACK_OFF` was
exact on 0/2, while the direct route stayed exact 1/1 in treatment and 1/1 with
the stack disabled. The direct control therefore separates recurrence from
ordinary symbol copying and emission. The unsupported cue selected one physical
explicit STOP. Every one of the seven treatment/control/probe traces had
physical custody. All seven neutral consequence resolutions recorded zero
changed edges and preserved the initial/final digest
`e122718efb8f419fd673cfb20098d0cfa06451001ba83669d93b93f6c97bd00a`.

The immutable artifact is
[`r4_recurrent_assembly_stack_gate_v1.json`](../../benchmarks/r4_recurrent_assembly_stack_gate_v1.json),
SHA-256
`11dbbbd205c2ba1f59d8c2369a1190f5dc6f232205e425f3be07c951c579912a`.
Fixture-content, fixture-source, runner-source, gate-spend model-source, and
cortex-source SHA-256 values are respectively
`e90933a769fcbb36a0ec8c316d4ee197c3590e4641bf2082e19007b15878d3ef`,
`bbe6f43ada638b96e2d1062688169c5555e99e28441807af59be11c20cae7db1`,
`de927bba97950e528ffeaa949b804d4045a74fdc4fcb7a0f94c8595fb138e4f6`,
`242cbbea8ff84f8ae8baedf5780e3a0bc346f07ad56fe966df5c9efde4d53f7e`,
and
`9cf37fc36978c3389a86a2856401f6c528d1a95f491dab0eb1b8662edcd055db`.
Independent read-only audit reproduced the static fixture and public-training
custody, scorer token IDs, emitted-byte hashes, raw physical trace summaries,
trace hashes, decision-freeze hash, neutral digests, and all score counts
without importing or rerunning the learner or gate.

This result is public developmental causal mechanism fit only. The scorer was
visible during implementation, so the artifact truthfully records
`cohort_novel_at_artifact_spend=false`. Model-training-unseen combinations and
symbols do not restore developer-heldout novelty. The result is not evidence of
continual improvement, a mesoscopic forcing-gate pass, repository competence,
protected transfer, SWE performance, or broad intelligence.

The gate-spend mechanism accepted at most five examples and rebuilt its full
program set after every complete success. It therefore did not solve the
incremental no-replay requirement or connect assemblies to the persistent
coding loop. The successor incremental implementation is in progress. Its next
step is a target-free experience adapter, followed by the already-frozen fresh
V6 integrated stream rather than another opaque mechanism fixture.

V6 has sixteen public-TRAIN tasks: stage counts 4/4/4/2/2 for renamed transfer,
nested multi-fragment recombination, execution-feedback revision, delayed
retention, and insufficient evidence. Arities 1, 2, 4, and 5 occur four times
each; fourteen tasks expect execution and two expect immediate STOP. Its
task-source, train-stream, and stream-custody SHA-256 values are
`34a19e89e8893f2a496cda422e7e2d5693c7d65cb68a4ebc1a0699e8b9049e24`,
`4291faf1c0035dbff79b40250ab8cfd0e718d0a610ffd73684481a93fed60303`,
and
`e19d14234b9d3bef90927e176590ede49f17c8280208712ee0a455b743cf672b`.
The integrated cohort was subsequently spent exactly once.

### Fresh recurrent-assembly continual V6 session

The frozen 16-task public-TRAIN V6 stream evaluated the incremental
`FactorizedFragmentMemory`, `RecurrentAssemblyStack`, and
`AssemblyFactorizedPointerModel` behind the same persistent coding boundary.
Treatment, an independently grown no-update control, and an independently grown
`STACK_OFF` first-attempt arm began from equal public-prior state. The runner
sealed its 20-file dependency manifest and exclusive reservation before the
first episode, then wrote one terminal artifact atomically.

The forcing gate failed: treatment/no-update/`STACK_OFF` scored 1/16, 1/16,
and 2/16. All successes were correct uncertainty STOPs. The fourteen executable
tasks scored zero in every stage: renamed transfer 0/4, nested multi-fragment
recombination 0/4, execution-feedback revision 0/4, and delayed retention 0/2.
Correction, causal retention, and recurrence event sets were empty. The public
prior remained exactly 8/8 in both main arms.

The failure distribution localized a representation regression before program
semantics. Each main arm made 37 attempts: six physical STOPs, fifteen
host-rejected pointer programs, and sixteen materialized programs rejected for
signature mismatch. Nine pointer failures referenced index 2 with a two-role
inventory and six referenced index 3 with a three-role inventory. Zero attempts
invoked the sandbox. The fourteen executable first attempts split into four
physical STOPs, four pointer rejects, and six signature rejects.

Treatment nevertheless applied 31 negative updates and changed its digest.
Twenty-seven updates hit one projected span and four hit another. Across all 37
aligned main-arm attempts, however, treatment and no-update had identical
tokens, physical trajectories, structural edges, STOP choices, raw pointer and
materialized hashes, host outcomes, retry classes, and scores. Every non-STOP
pointer output was byte-identical to one of the eight public-prior outputs; only
five stored programs were ever selected and one supplied 18/31 outputs.

Direct source inspection explains the evidence. Row-support factor refinements
couple downstream slot cohorts to one training owner. When those refinements are
active, projected slot races ignore plasticity ticks. Negative consequence
resolution changes only the last structural edge, which is usually a shared
anchor rather than the role/chunk choice responsible for the complete failed
proposal. The architecture can traverse one coherent stored program, but it
cannot yet recruit role-relative chunks across owners into a novel program for
plasticity to select.

The artifact is
[`r4_integrated_continual_coding_session_v6.json`](../../benchmarks/r4_integrated_continual_coding_session_v6.json),
33,242,148 bytes, SHA-256
`73d3348d44a5910a226a3c9405db2e6d31381ed25934d3a74996624c377f9b58`.
Task-source, train-stream, and stream-custody SHA-256 values are respectively
`34a19e89e8893f2a496cda422e7e2d5693c7d65cb68a4ebc1a0699e8b9049e24`,
`4291faf1c0035dbff79b40250ab8cfd0e718d0a610ffd73684481a93fed60303`,
and
`e19d14234b9d3bef90927e176590ede49f17c8280208712ee0a455b743cf672b`.
Artifact/reservation custody matches, all 90 decisions have terminal resolution
records, and no digest-chain discontinuity was found. V6 is immutable and must
not be rerun or tuned. Its claim is fresh public-TRAIN falsification and failure
localization only—not continual improvement, mesoscopic gate completion,
protected transfer, Simple-SWE, SWE-bench, Gemma, Opus, or intelligence.

The licensed successor was one learned role-relative sequence/chunk compositor
with physical current-role traversal, cross-owner transitions, STOP, and a
selected-chain eligibility trace. A small public novel-output causal probe had
to show a pointer program absent from stored full outputs and lose it under the
compositor/recurrence ablation; failure inhibition had to attach to the selected
role/chunk binding rather than globally depress a shared suffix.

### Role-relative compositor successor and terminal V7 session

The immutable exactly-once
`PASS_R4_ROLE_RELATIVE_COMPOSITOR_GATE_V1` artifact passed all 12 checks. Eleven
complete opaque demonstrations grew 36 cells and produced a novel six-emission
composition across two contexts with fresh-role binding. `COMPOSITOR_OFF` and
`RECURRENCE_OFF` each caused physical STOP; ambiguous and unknown inputs also
stopped. Exact-once consequence resolution and neutral digest identity held. A
target-free negative receipt changed the selected GID from 6 to 7, while
`NOGO_OFF` repeated GID 6. The artifact is
[`r4_role_relative_compositor_gate_v1.json`](../../benchmarks/r4_role_relative_compositor_gate_v1.json),
SHA-256
`921f78601a9e1e897ed16f2472227592ce3b35ec17feb76836899949df3e679c`.
Runner, compositor-source, and fixture-payload SHA-256 values are respectively
`9f4a745ef900d0139f2a1d2cd6b07d0cdba2591b975ea8e670a436d79f8c8b98`,
`26aa90d8b6b56a9ffb3094bbf45b93dd926d1d6e8e2ee33610ed00883a11e87d`,
and
`5c847e0c942097b3d2413149523dfd28a346d5c64ff0b62884e196800fc24972`.
The claim is deliberately limited to an opaque-byte causal mechanism. Complete
normalized demonstrations and learned contexts remain supplied; this is not
coding, integrated continual learning, transfer, SWE, or intelligence evidence.

The production integration then preserved a single selector. Complete
pointerized outputs discover repeated inventory-to-end runs and collapse them
to internal role markers. `FactorizedFragmentMemory` seals role-loop and
root-edge custody; `RecurrentAssemblyStack` physically role-binds, races
`CONTINUE` against `EXIT`, retains one pending receipt, and applies
prefix-qualified chain-local NoGo to the selected proposal, including
active-refinement scores. Focused development passed 7 assembly cases and 9
recurrent-stack cases and retained public-prior replay at 8/8. It emitted novel
size-three `PACK[<@0>](<@1>, <@2>)` from size-four training, lost it under
`STACK_OFF`, and changed the next proposal after a scalar negative receipt.
Root combined verification was 18 passed and targeted Ruff was clean. This is
implementation and focused developmental evidence only, not an integrated
coding-session or SWE result.

The planning and live paths still traverse twice and preseed future factors. A
single-pass attempt broke public replay and was rejected, revealing hidden
whole-row coherence dependence. That architecture debt remains explicit.
Separately, `DendriticCortex.race_currents_first_spike` now provides the stack
an event-exact compiled first-spike race. For 16 candidates repeated 200 times,
the measured call fell from 0.67846 seconds to 0.00693 seconds (about 97.9x)
with an identical winner and spike time; simulated event steps fell from 600 to
52. This is optimization evidence, not capability evidence.

Development returned immediately to the frozen V7 integrated stream rather
than opening another microgate. It contains 16 public-TRAIN tasks with stage
counts 4/4/4/2/2 and interface arities 1/2/3/5. Task-logical, masked-task-source,
and custody SHA-256 values are respectively
`63227e99f894654a32873b100682f30baa786d3a5e06018d2083f1b58820a117`,
`a73bdc1fb15fceccdc2419ca18a279de0d023a57e4ce4c39497117f637c8f2cc`,
and
`25d7db8cb294f5aec1f5953a336b1ada1bd36917fe0e7052081b86bfb352f35e`.

The session was spent exactly once. Treatment/no-update/`STACK_OFF` scored
2/16, 0/16, and 2/16. Every success was one of the two correct physical STOPs;
all fourteen executable tasks failed in every arm. Treatment made three failed
program attempts, applied three negative updates, and recorded sixteen physical
STOPs; after its third update, every first attempt for tasks 4–15 stopped.
`STACK_OFF` stopped on all sixteen. The no-update arm made
42 non-STOP attempts: five distinct raw pointer programs became fifteen
materialized sources, 36 entered the sandbox, and all failed. This is a real
advance beyond V6's zero sandbox entries and zero novel raw programs, but it is
not a solved task or a passed integrated gate.

Live role binding caused 30/42 no-update raw proposals to differ from byte-exact
prior programs. The mechanism therefore transfers structural interfaces, while
the finite owner set still replays familiar semantic bodies. Negative feedback
rotates through existing choices and collapses to STOP without recruiting a new
semantic operation. Treatment public-prior replay fell from 8/8 to 6/8, while
no-update remained 8/8. STOP was nonselective: treatment stopped almost
everywhere, `STACK_OFF` stopped everywhere, and no-update executed on both
intentionally ambiguous tasks.

The terminal artifact is
[`r4_integrated_continual_coding_session_v7.json`](../../benchmarks/r4_integrated_continual_coding_session_v7.json),
30,296,471 bytes, SHA-256
`8ce4e91d10b4028d1debe38aa96662211c76940ae08f002f725a5e417538329f`.
Runner and raw task-source SHA-256 values are
`e83696896ac658a8e19a6fdfa283638aa11d2cb67578343a1be7f97cb6eea8f7`
and
`4e3f3f9cc1ccaf20f7b5b0fda1730e5a7fe532f7c11e3ef0dafdd5327970d8b0`.
Reservation/final custody matches; 79 decisions each have one terminal
resolution, 45 authenticated resolution receipts were recorded, only the three
treatment negatives changed learned state, initial roots match, and both
no-update digests stayed unchanged. V7 is immutable and must not be rerun or
tuned.

The licensed successor is a learned semantic next-chunk graph/global workspace
that can recruit reusable fragments from successful experience, order them
across owners through recurrent cell decisions, preserve sequential causal
state without future-factor preseed, and place negative eligibility on the
responsible local transition. It must grow alternatives rather than only
exhaust a fixed proposal set. The claim remains fresh public-TRAIN
falsification/localization only—not continual improvement, mesoscopic gate
completion, protected transfer, Simple-SWE, SWE-bench, Gemma, Opus, or
intelligence. R5/R6 remain sealed.

### Sparse causal-sequence successor and terminal V1 microgate

The next-chunk hypothesis became `CausalSequencePointerModel`. Successful
episodes grow exact episodic traces plus sparse sensory, position, and recurrent
postings. Only active learned cells join each semantic race alongside tonic
STOP. Repeated pointer runs grow role-start, generic successor-offset, and
live-successor-conditioned `CONTINUE`/`EXIT` distributions, so role binding is
also a learned physical race. Successful emitted trajectories can grow online;
one pending failed trajectory applies state/step/predecessor/token-local NoGo.
`RECURRENCE_OFF` removes recurrent postings without bypassing the normal race.

An independent audit found and closed pre-spend ceremonial-role, broad-ablation,
dense-candidate, host-STOP, answer-bearing-feedback, runner/API, and dependency-
custody defects. Six focused model checks plus three frozen-fixture checks then
passed. The exact source was committed at `5f0f89e`; model and runner SHA-256
values were
`0c7eb90d253943b13f8670696224ef9747859c0fc44daf374e2685a8cd6a5116`
and
`f1f17b79db185d984c65c40d44eb94dcc447a4bff5aa94f76c3d44f47443c042`.

The independently authored three-demonstration opaque composition cohort was
then spent exactly once. It terminally wrote
`FAIL_R4_CAUSAL_SEQUENCE_COMPOSER_GATE_V1`. The first treatment completion
physically selected STOP on the fresh held-out cue combination and therefore
left no pending action. The next call did not produce the expected pending-
action exception, so the runner terminated before scoring the later ablation,
ambiguity, negative-revision, and retention predicates. The failure artifact's
result digest is valid, but its terminal exception record omitted the first
trace. That observability defect prevents localization to a precise step or
candidate; it did not cause the physical abstention.

The immutable artifact is
[`r4_causal_sequence_composer_gate_v1.json`](../../benchmarks/r4_causal_sequence_composer_gate_v1.json),
2,531 bytes, SHA-256
`6f5042e1ffaa22b6cac563c565f677330ac84ce0ee3ce66379b9f10d2497c116`.
Fixture source, payload, and custody roots are
`135e522aa08bd568237a3414da3993ddc637b397c0e7c654fb8251faa15cabe8`,
`28a0801a7e9b50e6235c6a793c9481c186e056ba0e8b48a3c332e31539c18e56`,
and
`c98abb519bbd6a4cc81b291cb1c45d39f21b991752353cdd2f980f4a5af40439`;
the expanded behavior-source custody root is
`1f7e669c867d164a93761cec1ffc4044e73bdb04e64fef9133b761ed94074d3e`.

Static source forensics localized the leading interaction. Repeated chunks
became single phrase tokens while singleton desired chunks stayed multi-byte;
the per-target two-context gate suppressed the singleton mappings, and
absolute token steps misaligned unequal-length segments. The synthetic tests
duplicated observations and used fixed-width atoms, hiding both constraints.
Lowering the support threshold to one would merely admit accidental singleton
correlations. The broader successor is dual-path promotion: retain repeated-
context evidence, and add contrast-certified cue-region-to-output-residual
relations learned from matched episodes whose shared background cancels.
Stable byte time plus learned segment cells should replace frequency-dependent
absolute token positions.

V1 is immutable and must not be rerun or tuned. Per the roadmap's integrated
bias, the redesigned V8 workload comes before another microgate. Only if its
failure distribution confirms this bottleneck should a fresh, model-blind
three-factor corner cohort test the contrastive relation path. No protected,
SWE, Gemma, Opus, or intelligence claim is licensed; R5/R6 remain sealed.

### Terminal V8 integrated session and action-boundary localization

V8 replaced the synthetic composition gate with a larger persistent
public-TRAIN repository stream. Twelve fresh tasks covered acquisition,
cross-source composition, consequence-driven revision, delayed transfer, and
two insufficient-evidence controls. The one public action surface was a strict
generic `submit_patch` JSON-Schema affordance; private evaluation stayed in an
isolated worker. The treatment, no-update, and `CONTRAST_OFF` arms were grown
independently from the same eight public successes, and all posthoc scoring was
deferred until every arm had completed.

The terminal scores were 2/12, 2/12, and 2/12. Only the two correct STOP tasks
passed. Across 48 attempts, 32 selected physical STOP and 16 produced a
materialized `<tool>` response that failed canonical JSON/tool-shape decoding.
No action was admitted, no patch was extracted, no sandbox ran, and no semantic
repository judgment occurred. Treatment applied four digest-changing negative
updates and turned each malformed proposal into STOP on retry. The no-update
arm repeated each proposal identically; `CONTRAST_OFF` always stopped. This is
causal evidence for failure-conditioned local suppression, but not for
correction.

All malformed attempts shared pointer-output SHA-256
`54e41645aad6fcca453a1cdb543c94115a5c25fb33e9aae0db419f59d1354c97`
while producing four distinct prompt-conditioned raw hashes. Thus a whole
contrast-selected prior pointer template crossed lexical inventories and
corrupted the action shape. Separately, the unchanged bootstrap checkpoint
replayed only cases 0, 3, and 7 of eight; all arms physically stopped on the
same other five. The next work is licensed at the representation boundary,
before semantic patch synthesis: exact observations need a dedicated neuronal
episode assembly, and interface serialization needs a generic stable codec
whose structure cannot be rebound as prompt-local vocabulary. Payload content
and all cognitive choices remain neuronal.

The spent artifact is
[`r4_integrated_continual_coding_session_v8.json`](../../benchmarks/r4_integrated_continual_coding_session_v8.json),
SHA-256
`adee84be37609f03eca023817571b53a434662aee4f5455a90d9ae3a18ccb9b3`,
with valid embedded result digest
`3d8b50e6903bcfe679cc2a584bec8cc49fc16b9a0fa0c25feaae73feb5d105a2`.
Its task, manifest, and runner sources remain frozen. V8 must not be rerun or
tuned. This is pre-execution failure localization only; it is not software
competence, continual improvement, protected transfer, SWE-bench, Gemma,
Opus, or intelligence evidence. R5/R6 remain sealed.

The localized representation correction was then implemented without another
adjudicative microgate. Commit `d177565` added a learned exact-state episode
posting: an episode-template cell wins a first-spike race, its literal
tokenizer atoms replay through successive physical races, and physical EOS
closes the trajectory. Exact negative credit attaches to that episode cell;
`EXACT_ASSEMBLY_OFF` removes the route and produces tonic STOP without changing
the learned digest. Commit `0cd20e7` added a generic stable tool transport:
neurons emit a numeric action wire containing action-name/key UTF-8 bytes and
normal pointerizable payload values, while a deterministic environment adapter
only canonicalizes the emitted structure into `<tool>` JSON.

Focused modules passed 19 and 20 tests. A read-only diagnostic over the eight
public bootstrap actions—not the V8 task stream—measured 8/8 byte-exact replay
and 8/8 valid `submit_patch` decisions, versus 8/8 physical STOP under
`EXACT_ASSEMBLY_OFF`; initial and post-neutral digests were identical. Normal
inference ranged from 19.5 to 33.8 ms per action. This repairs the exact-memory
and interface-transport boundaries exposed by V8 but says nothing yet about
novel patch semantics. The next evidence comes from a separately authored,
fresh V9 integrated TRAIN stream.

A subsequent read-only scale characterization used punctuation-only exact
chains so tokenizer atom counts were unambiguous. Median replay was 62.1 ms for
256 atoms, 93.4 ms for 1,024, and 283.7 ms for 4,096; growth took 0.414, 1.789,
and 6.777 seconds respectively. At 4,096 atoms, repeated direct first-spike
calls accounted for 230.2 ms, about 81% of total replay, and traced peak Python
allocation was only 2.339 MiB. This localizes a performance rather than a
cognitive failure: a future compiled sequential exact-chain race can amortize
the call boundary if it preserves winner, spike-time, tonic-STOP, EOS, and
dependency parity. It should follow rather than perturb the frozen V9 source
evaluation.

### Terminal V9 fresh integrated spend and transport exception boundary

The successor architecture was frozen before a separate author produced the V9
cohort. Commit `73d6aa8` then froze twelve fresh public-TRAIN tasks, their
manifest, and the hardened three-arm session runner. Fixture, manifest, and
runner SHA-256 values are
`2ff864ec3819e79dca5ef11dbd7ebe2a0b1dee2fe50bc47c65d424d3c8b5ef5d`,
`b01d00a2ba26c3245ca39251a024db6764679533da15c8ab3bc00d17d8a06ddc`,
and
`6964c2be771b11c3eaadcd2271f7b1ec240636e61f688b84e152abcd5431262d`.
The logical TRAIN-stream commitment is
`72d246ec3d35abdbb3519bead1b2c1329d5cd36bfe838ba00cd170f4a707a124`,
its custody commitment is
`a060dbb63fae53ebb0d295c520ed81ef7a24d81ac6c396cb4151f7a5914b71de`,
and the authenticated public/source preflight digest is
`e03dccacb2b4962b908b113149dead1d5da457760d5a1d25000e6e995cf270b2`.

The exactly-once run reserved its artifact and grew treatment, no-update, and
`CONTRAST_OFF` from equal initial learned digest
`f1d46408f0a7333eb8d6c14675917cf46e57dd8a3c0ce2fb680cfc3b05a429cc`.
Pre-retention passed. During the first treatment task's first episode, a
recognized malformed neuronal action transport raised `NeuronalToolModelError`
out of `infer`. `_run_task` never returned, so no task record completed. The
runner terminated `FAILED` before complete arm streams and before the private
posthoc scorer could open. No semantic patch, evaluator-reach, pass-rate,
recovery, or continual-improvement conclusion is available from V9.

The spent artifact is
[`r4_integrated_continual_coding_session_v9.json`](../../benchmarks/r4_integrated_continual_coding_session_v9.json),
SHA-256
`3dc0b74367b0fc2ae979e4e8d7e9ba73644b010bc717ec7ce117b417c0ba9737`,
with embedded failure digest
`65e0e95856636140e07ff8077901c54b318c582e595d1ed28fa0827126d324f1`.
Artifact commit `2e0e655` preserves the terminal record. V9 is immutable and must
not be rerun or tuned.

This localizes the next general correction to the action interface rather than
repository semantics. A recognized malformed transport must fail closed as a
normal generation failure, so the ordinary consequence path can learn from it
without a host adapter selecting an alternative action, tool, or payload. The
corrected source receives a fresh V10 cohort and exactly-once spend. V9 licenses
no protected, SWE, Gemma, Opus, or intelligence claim; R5/R6 remain sealed.

The successor used a fast synthetic reproduction rather than reopening V9. A
fresh model learned the incomplete recognized wire `[[84,79,79,76,1]]`; real
`infer` reproduced the same typed exception twice. Four hypotheses were ranked,
and the first was confirmed: decoding already occurred after provisional
emission, pending-query, and trace custody were sealed, but its typed failure
branch re-raised instead of returning that fail-closed state. Commit `f3dbaa8`
changes only that `raise` to `return provisional`. Decoder strictness is
unchanged, the emission remains a neuron-selected non-STOP generation,
`invalid_action_transport` remains causal trace evidence, and the pending choice
can receive normal negative consequence credit. The minimized regression and
20 focused neuronal-tool tests pass. Source and test SHA-256 values are
`25aedcdf2a1868f7e7b0950b29956a1b89c0e6f9d533795eb26e08503d0c010b`
and
`0771a77e822962d33cf3c9315daf7d1065fc6eff9ed35a4ec4daa66a4ddd5f39`.
This is runtime-continuity evidence, not semantic coding evidence. It freezes the
architecture boundary before fresh V10 task authoring and does not authorize a
V9 rerun.

### Terminal V10 complete integrated spend and relational-action boundary

After commit `f3dbaa8` froze failure normalization, an independent author built
a fresh V10 public-TRAIN cohort without reopening V9. Its twelve tasks were
ordered across four direct repository relations, four delayed
history-required transfers, two consequence-guided revisions, and two
genuinely ambiguous STOP cases. Treatment, independently grown no-update, and
`CONTRAST_OFF` began from the same learned digest
`f1d46408f0a7333eb8d6c14675917cf46e57dd8a3c0ce2fb680cfc3b05a429cc`.
The exactly-once session reached terminal infrastructure status `COMPLETE`: all
36 task records, all 39 attempt resolutions, post-retention probes, and posthoc
scores exist.

All three arms scored 2/12, driven only by correct physical STOP on the two
genuinely ambiguous tasks. Direct repository relation was 0/4,
history-required transfer was 0/4, and consequence-guided revision was 0/2 in
each arm. No sandbox evaluator was reached and semantic pass was zero. All four
non-STOP attempts occurred only on stream ordinal 8,
`train-v10-lab-protocols-92`; each carried the identical 301-byte action and
pointer-output SHA-256
`d82acab72169c718948acd1aa9f5b4320c3414545f0e5b02d0b0f350ea2675d3`,
which exactly matches retained public-prior probe index 4. Treatment used 13
attempts: that valid canonical transported action reached the patch validator
once, was rejected before a fixture receipt/evaluator, received one
authenticated negative update, and was followed by STOP. Its final learned
digest changed to
`337d57517d6b3d66ab696f526936398ac84a091cfd303645568beb4ba6f94846`,
but that plasticity produced no semantic pass. No-update used 14 attempts and
repeated the same valid transported action three times; all three were rejected
by the patch validator before evaluator execution. `CONTRAST_OFF` used twelve
attempts and selected STOP on all twelve tasks.

Every arm retained all eight public priors both before and after the task
stream. The terminal evidence therefore preserves exact episodic replay and
stable action transport, and identifies the only active proposal as exact prior
pointer-template replay without task-local structural binding. Local failure
credit is shown only to suppress that rejected proposal; it neither corrects
the action nor constructs a valid source-relative alternative. The broadly
reusable bottleneck is now **source-relative relational action transduction**:
experience must recruit relations among a current goal, current source
observations, and action segments, then bind those segments to the live source
without target spans, an AST/task ontology, or stored patch templates.

The spent artifact is
[`r4_integrated_continual_coding_session_v10.json`](../../benchmarks/r4_integrated_continual_coding_session_v10.json),
254,127 bytes, SHA-256
`08a3b8a121509977a9ffb990bb7b9a6a4eb3979128d7d0ff3a55bff148045852`,
with embedded result digest
`62dd215e991bf7621c53861b0bdb4322c2b99565abfd811442d95ba882f5a099`.
Artifact commit `a7002f3` preserves the terminal record. `COMPLETE` is an
infrastructure/custody result, not a passed integrated capability gate. V10 is
immutable and must not be rerun or tuned; it licenses no protected, Simple-SWE,
SWE-bench, Gemma, Opus, or intelligence claim. R5/R6 remain sealed.

### Raw-source motor stack and cue-free Python pilot

V10's source-relative relational-action failure was followed by a new,
unprotected construction lane rather than any V10 edit or rerun. Commit
`c267958` learned source-relative recurrent motor programs; `6a18ac2` compiled
successful token transformations into generic motor steps; `eab9e8d` learned
raw-tape role binding; and `18e0b7d` exposed a composed controller whose public
inference inputs are only context events and a complete source tape. The
training compiler still observes a successful before/after trajectory, so this
sequence banks a mechanism rather than autonomous program induction.

Commit `0c4740b` then ran a public-TRAIN whole-workspace pilot. Commit `8d0a698`
subsequently hardened its evaluator to reject path-only/empty-source actions and
to require exact observable program output rather than exit status alone. Each
task made one controller query over all issue tokens and every sorted repository
file.
Three fresh same-layout tasks with renamed identifiers, different paths, shifted
positions, and non-source distractors failed real Python before the edit.
Treatment emitted one live path plus a complete replacement composed only of
token occurrences in the current workspace tape; all three passed real Python
afterward. A fresh two-Python-file ambiguity collision-STOPped without editing.
`BINDER_OFF`
emitted nothing and binder-STOPped all four tasks. `TAPE_OFF` emitted nothing,
tape-STOPped the three action tasks, and preserved the ambiguity binder STOP.

The source-tape hot path was optimized without altering its scientific
semantics. Commit `a99473b` measured a 26.2x batched first-spike improvement on
4,096 independent rows. Commit `73929cf` added exact inverted occurrence-
neighborhood scoring, and `2693901` integrated it into the binder, reducing an
actual 4,096-token two-role query from 12.002 ms to 7.663 ms (1.57x).

These results close only the narrow renamed same-layout exact-lexical
source-grounding question. They do not show learned issue-to-file applicability,
continual learning, semantic issue understanding, arbitrary editing, recovery,
protected transfer, or model-level coding competence. The next causal target is
an immutable replay-free successor learned from an authenticated execution
success, followed by a later fresh task that treatment solves while a persistent
no-update control STOPs. R5/R6 remain sealed.

Commit `88049e3` implements that replay-free successor as frozen neuronal shards
behind one fitted-GLIF ensemble race. PASS receipts can be constructed only by
an authority path that invokes a caller-pinned evaluator, and each receipt binds
the exact parent checkpoint, observation, action, evaluator root, and outcome.
Seed and response paths isolate mutable pending state. One accepted experience
is compiled once; no prior experience is read or replayed, raw training
sequences are discarded, and retained event bytes are acknowledged as learned
synaptic content. Public structural quantities are capacity/support estimates,
not an asserted physical ledger.

The real-Python session in commit `d10e09c` starts treatment and `NO_UPDATE`
from identical roots. Both initially STOP on the new exact context. The same
executed correction receipt enters both arms: treatment appends a shard and
passes 2/2 later renamed/path-shifted repositories, while `NO_UPDATE` commits
zero state and physically STOPs 2/2. Base behavior is retained and mixed
contexts collision-STOP. The new shard reports estimated reservations of 4,387
cells and 61 synapses plus 110 support ticks. This is same-layout exact-context
continual motor recurrence with an external successful teaching trajectory, not
language learning, context abstraction, autonomous correction, or broad coding
competence.

Commit `ee5b8df` then froze the harder multi-source applicability probe before
any corrective mechanism was added. Two public issues each repeated one fresh
symbol from one of three matched-layout Python files; a third issue repeated two
symbols and was genuinely ambiguous. All three inference calls received one
full issue stream plus one full workspace tape exactly once. The current
controller emitted nothing: both uniquely named action tasks binder-STOPped and
the ambiguity correctly STOPped. This cleanly localizes novel query/source event
equality followed by learned enclosing-segment role propagation, without a
target file, span, candidate action, or repaired fresh source.

The large binder-race preparation path was separately optimized in commit
`d64bbd9`. At the measured 128-candidate crossover it switches to vector bounds
validation and reuses the score-index array. Exact scalar parity holds at 127,
128, and 4,096 candidates; the isolated 4,096-candidate race fell from 604 to
471 microseconds (1.28x). This changes speed, not the negative multi-source
result or its scientific boundary.

Commit `8b54a4f` then banks the smallest generic computation proposed by that
failure. `CrossStreamSegmentBinder` learns, from complete opaque streams, a
unique query/source equality seed, a recurrent offset to the demonstrated role,
and a stable bounded containment envelope. It transfers to a fresh renamed
three-segment stream and physically STOPs on absent or duplicated equality,
broken containment, and equality/propagation ablations. This is micro-gate
evidence only. It does not reverse the frozen `ee5b8df` result, because the
current workspace transport still lacks a shared fine-grained identifier event
space and no multi-file motor action has yet been emitted or executed through
the new binder.

### Multiscale multi-source integration and boundary-position failure

Commit `2f6ee05` adds a lossless multiscale event tape. It preserves complete
query and workspace bytes, exposes byte-identical identifier atoms in one opaque
equality domain, and embeds live path/physical-line occurrences for the existing
motor stack. Commit `5ed2df1` composes that tape with cross-stream binders and
the source-relative action tape behind one complete-issue/complete-workspace
controller query.

Commit `8358b67` immutably spends the resulting V1 public-TRAIN cohort. Treatment
executes the exact intended repair on 1/2 fresh action tasks. The other action
physically STOPs in the first role binder, while the genuinely ambiguous issue
correctly STOPs without editing. `EQUALITY_OFF` STOPs 3/3 tasks, and
`PROPAGATION_OFF` STOPs 3/3 tasks. This is an honest integrated failure; the gate
did not pass.

Both demonstrations place their intended file in a middle workspace position.
The fixed-radius containment envelope therefore learns predecessor-segment
context. A fresh target in the first segment lacks that predecessor context and
cannot satisfy containment, so inference STOPs before actuation. V1 remains
unchanged and must not be rerun or tuned. The next causal seam is a
boundary-invariant opaque segment envelope, followed by its integration with
the persistent append-only learner.

Commit `7a747f5` subsequently banks the causal micro-gate.
`BoundaryAwareSegmentBinder` learns nearest recurring opaque left and right
delimiter relations from complete streams without delimiter or location routing
inputs. Fresh renamed targets at first, middle, and last segment positions all
BIND. Cross-segment ambiguity and a missing nearest boundary physically STOP;
`EQUALITY_OFF`, `PROPAGATION_OFF`, and `BOUNDARY_OFF` each physically STOP as
well. Five focused tests passed. This is mechanism evidence only and does not
revise the immutable V1 score. Its new integrated successor and genuinely fresh
cohort are still pending.

Commit `74045b2` then freezes that integrated successor controller. It combines
the `7a747f5` binder with the lossless multiscale event tape and the existing
occurrence-preserving action tape while retaining
`respond(query, files, *, ablation)` as the sole inference boundary, with
`files` holding the complete workspace. It adds the `BOUNDARY_OFF` causal arm.
Three focused composition tests passed fresh
synthetic first-, middle-, and last-position targets, genuine ambiguity, and
physical STOP under equality, propagation, and boundary ablation. This remains
mechanism/integration evidence, not a real-Python successor result.

Commit `f596fe1` next introduces authenticated replay-free multiscale growth as
immutable frozen controller shards. Its first version still named the legacy
fixed-containment controller and therefore was not yet the boundary-aware
successor required by V1's failure.

Commit `3aff79f` immutably spends the fresh boundary-aware V2 cohort exactly
once. Treatment performs the correct real-Python edit on 3/3 fresh repositories
with intended targets at sorted positions first, middle, and last. All three
fail before and produce exact expected output after; only the intended path
changes, decoys remain byte-identical, and emissions come from the current live
tape. The fourth, genuinely ambiguous issue physically STOPs at equality
collision. Equality-, propagation-, and boundary-off arms each emit 0/4 and
STOP 4/4. All inference is single-query, workspace transport is lossless, and
neutral checkpoints remain unchanged. The single-shot result record SHA-256 is
`7eac8918de381657149e9cfa0aee9ddac3d8ab677023a0cc6b1ede8cecefa07b`.

V2 closes the fixed-radius position dependence exposed by immutable V1, but no
more. It retains two teacher-visible successful trajectories, a fixed motor
program, exact lexical issue/path equality, and small matched-layout Python
repositories. It is not continual learning, operation induction, language
understanding, recovery, protected transfer, or broad coding capability.

Commit `41f6b8a` then deletes the obsolete online-growth coupling and retargets
the stack exclusively to the boundary-aware controller. Two focused tests pass.
One authenticated current experience contributes two training phases with no
parent replay and no retained raw full training sequences. The estimated
base/append/successor accounting is 4,381/+4,382/8,763 cells, 33/+35/68
synapses, 145/+194/339 learned-event bytes, and 64/+68/132 excitatory support
ticks; the appended 68 ticks are 34 coactivation plus 34 positive-success.

This is a replay-free one-update mechanism, not a passed persistent integrated
gate.

Commit `34a7f52` then immutably spends the first persistent two-update
multiscale Python session. Its single-run result SHA-256 is
`5b2b561b7d4675bb04dd153e220ac9a42899aa706c2161262bb33f6e3883d267`.
The authoritative result is `gate_passed=false` and 2/5 scored transfers, with
two exact acquisition retries explicitly excluded. Although the runner carries
an aspirational prose claim, it does not supersede the gate or score.

Both new path-depth tasks initially physically STOP. Their independently
authenticated successful transitions each APPEND a shard, and each exact retry
then passes. A fresh flat-path transfer passes before either update and a
distinct flat-path retention transfer passes after both. The fresh renamed
one-directory transfer fails after update one, a second fresh one-directory
transfer fails after update two, and the fresh two-directory transfer fails as
well. Genuine equality-collision ambiguity, the no-update branch, and equality,
propagation, and boundary controls physically STOP. Both append receipts report
zero parent experience replay and zero retained raw full training sequences.

Accounting begins at 4,381 cells, 33 synapses, 145 learned-event bytes, and 64
support ticks. Append one adds 4,382/35/198/68, split 34+34 across coactivation
and positive success. Append two adds 4,384/37/251/72, split 36+36. The final
three-segment learner (three frozen shards) reports 13,147 cells, 105 synapses,
594 learned-event bytes, and 204 excitatory support ticks. These remain explicit
estimates.

This failure localizes the difference between exact replay and transferable
schema growth. Each append duplicates one success to express two learning phases;
without independent variability, the binder treats a literal directory-prefix
propagation event as invariant and stores it. Fresh renamed paths at the same
depth then propagation-STOP. Variability-induced wildcard propagation learned
from genuinely independent examples is the next causal seam and is already
underway. V1 stays immutable; its correction requires a new successor and fresh
cohort. R5/R6 remain sealed.

Commit `ba7e016` then banks the localized variability-aware mechanism. From
distinct independent demonstrations, `VariabilityAwareSegmentBinder` separates
stable intermediate propagation events from wildcard positions. Fresh renamed
first-, middle-, and last-segment probes BIND. Stable-event corruption,
unsupported depth, equality ambiguity, missing boundary, final role collision,
and equality, propagation, and boundary ablations physically STOP. The focused
suite passes 6/6. This closes the isolated causal seam only; persistent
controller integration and a fresh cohort remain pending, and V1 stays failed.

Commit `2583bf8` follows with a scale-enabling inference optimization. The
multiscale stack prepares one immutable observation per response and broadcasts
it across its frozen shards rather than rebuilding it in each controller.
Controller/stack decision, trace, receipt, and checkpoint hashes and reported
`physical_cell_steps` remain exact; 7/7 focused tests pass. On the same synthetic
16-shard miss, median latency changes from 42.5321 ms to 16.5495 ms (2.57x),
while the conceptual neural work remains 8,352,600 cell steps. No spent benchmark
was rerun. The result changes runtime cost, not the scientific capability claim,
and makes larger persistent stacks more practical.

A later unfrozen engineering probe tested local fitted-GLIF batching in the
relational reader. Its generic 2,048-by-257-cell kernel improved from 64.436 ms
to 10.515 ms (6.13x) with exact race parity, but grouping and trace
reconstruction made the complete reader fixture regress from 569.466 ms to
587.880 ms while preserving its decision SHA, 7,175 traces, and 196,194,600
cell steps. The candidate was removed byte-for-byte. This negative result points
to batching complete causal frame-detector layers later, not accumulating local
batching wrappers; it changes neither production code nor capability evidence.

Commit `101edfb` then composes the variability-aware binder with the neuronal
source motor. Two independent full transitions train one shared action plus
stable/wildcard propagation. Physical fitted-GLIF races now also enforce role-
inventory and live-occurrence uniqueness; the focused suite passes 4/4.

The controller was frozen before V3 was authored at `d52f6b6`. The fresh cohort
was spent once and sealed at `3d9df99`; artifact SHA-256 is
`8d86fb1a25523f19cc4c0a7d7f2814c5e0978e36f4a9f8086b5d2d55243cafda`
and embedded result SHA-256 is
`fbec194680c82c088b5e23101d61da9f0ebe2ecf21b3d03dce42331901fe1243`.
Treatment passed all three fresh real-Python repairs at first/middle/last sorted
file positions plus the ambiguity STOP. Every live materialization changed only
the intended file, preserved decoys, and changed the exact failing execution to
the exact passing execution. Equality, propagation, and boundary ablations each
physically STOPped 3/3. All observations were lossless and single-shot, all
checkpoints remained fixed, and reported cell-step accounting was exact.

V3 closes the fresh variability/boundary composition seam only. Exact opaque
identifier equality and one prefix-truncation motor remain, fresh tasks do not
update the learner, and the persistent V1 gate stays failed at 2/5. The V3
runner, test, cohort, and artifact are immutable. That result localized the
coding successor to replay-free appending of generalized shards from independent
current successes.

Commit `fd38ef6` implements that successor mechanism. One immutable stack seeds
an already-generalized variability-aware controller and appends one new shard
only from an atomic 2–64-row cohort of distinct current observations, provenance
roots, and evaluator-issued PASS receipts bound to the exact parent checkpoint.
It compiles from the current cohort, stores custody hashes and learned state but
no raw rows, reports zero parent replay, and leaves both parent and checkpoint
unchanged on `NO_UPDATE`, duplicate, or conflict. At inference, all shards run
and a fitted-GLIF selector produces exactly one EMIT, tonic STOP, or a separate
collision-STOP relay. The focused 3/3 suite demonstrates deeper fresh transfer,
prior retention, parent immutability, and equality/propagation/boundary controls.
This does not revise V1 before a fresh integrated spend; evaluator-supplied
success, exact structural shards, and per-shard observation reconstruction are
still limitations.

The first successor cohort was frozen at `24ba8b9` and spent once. It reached
zero tasks: seed-stack construction raised `ValueError` because the authored
training streams did not meet the existing requirement of exactly one
query/source coincidence. Commit `05bc9af` seals the failure artifact (SHA-256
`23e954fa88c0ef61d205be31c6ae5fa4c31b0c9d67af68f706b5218896e98955`).
This is neither a positive nor negative learning result; it localizes a public-
experience/controller interface mismatch. V2 remains immutable, and a fix must
use a fresh cohort after validating that contract without opening its outcomes.

In the co-equal language lane, `442fb83` adds a lossless opaque multi-document
event tape with provenance, sentence structure, and distinct repeated token
occurrences but no semantic or coding ontology. Its focused suite passes 5/5;
ASCII lexical identities are the current bounded transport surface.

Commit `c999ea2` adds an answer-supervised exact-surface two-hop reader scaffold.
An independent audit found that its host computed frame and equality Booleans
before GLIF arbitration, so no neuronal-language claim was banked from that
version. Commit `7819520` closes that inference-execution defect: raw byte and
length sensory sheets drive fitted-GLIF coincidence cells, physical feature AND
and learned-frame OR, query-to-document and recurrent equality joins, document
inhibition, and a live-occurrence motor. Fresh entity recombination, learned
forward/reverse surfaces, document/position changes, distractors, ambiguity,
missing evidence, a one-byte query mutation, and four ablations remain causal.

The boundary stays explicit. Training still uses an observed answer and host
compiler to find one unique proof and exact one-slot/two-hop surfaces; inference
still host-enumerates ordered token-pair candidates. Thus `7819520` is neuronal
execution of a learned exact-surface schema, not predicate discovery,
paraphrase understanding, or open-ended reading.

Commit `fde2ecb` adds the adjacent domain-neutral binding substrate. Opaque
bytes allocate one active cell in each of 64 blocks, fixed modular/product
wiring supplies ordered reversible role/filler composition, an 8,192-byte
synaptic state superposes noisy prototypes, and fitted-GLIF first-spike races
perform transforms, cleanup, tonic STOP, and collision STOP. Its focused 6/6
suite covers ordered distinction, nested unbinding, fresh identities, noisy
cleanup, ambiguity/missing support, and causal controls. It contains no language,
code, relation, or answer ontology. It is mechanism evidence only: learned
surface-to-latent mappings, multi-item episodic retrieval, provenance-aware
revision, and persistent 1–3-hop language use remain the next integration.

Commit `f92dd3d` adds the first persistent successor. One raw
`RelationalReadingSuccess` advances an immutable checkpoint without replaying
parent episodes or retaining raw rows/documents. An exact surface promotes only
after two independent filler pairs, success provenances, document identities,
document provenances, and cell-bound context pointers. Query/document surface
matches, the one-hop subject join, unique-answer selection, live occurrence
motor, ambiguity, and all STOPs execute through fitted-cell races. The focused
1/1 fixture covers two interleaved surfaces, delayed fresh-filler transfer,
same-document rejection, missing and ambiguous evidence, parent immutability,
and five causal ablations.

Commit `7e988a9` adds a second persistent step: one exact surface can become
associated with another only when its complete ordered subject-to-answer filler
pointer physically wins cleanup against a prior episode. The learner stores two
ordered surface-link pointers. Frozen inference physically binds query/document
surface pointers, cleanup-selects learned links, conjoins frame and subject
signals, OR-collapses paths per live edge, and retains unique-answer/live-motor
arbitration. The focused suite passes 2/2, including same-subject, same-answer,
reversed-binding, same-surface, distractor, ambiguity, and causal controls.

Commit `a2521bb` separately composes the existing request/document and workspace
event tapes with ordered raw tool events behind one modality-complete sensory
surface. Exact bytes, provenance, presentation order, repeated occurrences, and
round trips are preserved; no task or answer ontology is added. This is a
lossless adapter, not intelligence.

Commit `3448797` adds neutral document acquisition after a frame has been
learned. Consolidation visits every sentence-local ordered edge and commits all
physical learned-frame winners, rather than a host-selected target. It retains
sparse surface/subject keys, key/answer fact assemblies, source provenance and
occurrence digests, plus one deduplicated counted destination lexeme. With the
source documents absent, fitted-cell relation/key/fact/lexeme/motor stages recall
two fresh facts; duplicate facts merge sources and a conflicting answer reaches
collision STOP. Three focused fixtures pass, including consolidation/retrieval
ablations, immutable parents, zero raw document/row retention, and exact repeated-
source binding-work accounting.

Commit `b5ccc73` adds a focused knowledge-to-action integration seam. Two
independent evaluator-authenticated before/after workspace transitions train
one factor-zero-to-occurrence relation, but the frozen checkpoint retains only
their evidence roots and learned neuronal structure. At inference the only
public input is a unified opaque request/document/workspace/tool tape plus an
ablation. The bridge invokes the persistent reader internally, seals its
physical retained-fact winner into a private pointer broadcast, unbinds and
cleans the anchor and lexeme factors, grounds the anchor against live workspace
occurrences through the variability-aware binder, runs fixed neuronal
`SUBSTITUTE -> END`, and resolves EDIT or STOP through one invariant physical
population. A single compact fixture passes fresh matched file/value transfer
with the emitted value absent from live sensory input, decoys, duplicate and
missing anchors, conflicting memory, and eight causal controls.

This is not a new integrated benchmark result. Full-workspace differencing is
training scaffolding, the motor program is fixed rather than learned, host code
still tokenizes and materializes the selected byte substitution, and the
upstream frames remain answer-supervised exact ASCII surfaces. The result closes
only the first retained-fact-to-occurrence seam.

Commit `955b32e` removes the observed-answer compiler from a separate, neutral
surface-learning path. Its immutable learner consolidates only raw opaque
documents. It proposes every two-position leave-out frame, but promotion
requires physical provenance, pair, and both-slot variability support
populations followed by a conjunction. Cross-frame SAME and SWAPPED hypotheses
each require two shared pair pointers and independent source support before
competing in one physical orientation population. A winner grows two reversible
frame routes. Commit `f11a5cd` adds an independently relabeled copy of the same
focused evidence.

Commit `4f23658` then renames the owner to
`FrozenNeutralNeuronalDocumentLearner` in
`neutral_neuronal_document_learner.py` and adds delayed fact memory without
widening the public seam. A new episode may consolidate facts only through
frames promoted in the parent checkpoint. Each eligible occurrence passes a
physical write-versus-tonic race, then stores two-way filler bindings, exact
endpoint lexemes, and opaque provenance/document/occurrence roots. Raw documents
and complete sentences are not retained. Identical facts merge provenance;
conflicting endpoint pairs stay separate.

When documents are later absent, fact-frame cleanup supplies the existing
route/orientation/alignment/bind-unbind/output path. Supporting provenance roots
remain visible as digests, while each distinct completion receives the same
motor drive; incompatible memories therefore collision-STOP instead of using
source count, append order, or a host truth rule. `FACT_CONSOLIDATION_OFF` and
`FACT_RETRIEVAL_OFF` causally remove write and delayed read. The focused 1/1
fixture passes live prequential transfer, delayed two-source recall in both slot
directions, parent immutability, retained conflict STOP, relabeling, ordering,
shuffled bindings, and the earlier structural controls. Descriptor-key postings
now restrict cleanup to all matching frames without selecting among collisions.

This banks exact positional two-slot structural recurrence plus persistent
provenance-bearing fact recall, not broad semantics or language understanding.
Tokenization and positional candidate enumeration remain host scaffolding;
similarity-based paraphrase, truth learning, provenance revision, and recurrent
multi-hop composition remain open. The older knowledge-to-action path still has
answer-supervised upstream frames. No integrated benchmark was run. The planned
balanced evidence-to-action session keeps those missing capabilities prerequisite
for both language and code outcomes rather than allowing coding throughput to
stand in for understanding.

These are public TRAIN-side development lanes; R5/R6 remain sealed.

## 13. Retired directions

Do not reopen these without a new falsifiable rationale and a newly named gate:

- ridge/readout retuning on the old feature space;
- static whole-prompt decoders;
- whole-program lookup/continuation as the composition mechanism;
- blind edit search and the failed persistent babbling scheduler;
- greedy single-step actor as a planner;
- task-specific H=2 prompt-projection patches;
- raising spent work bounds to rescue failed canonical tasks;
- restoring task/family/ordinal IDs to manufacture prompt uniqueness;
- leaking hidden property/DEV outcomes back into selection or architecture
  design.

## 14. Why this history is separate

The main roadmap should answer:

1. What are we trying to build?
2. What design principles constrain it?
3. Where are we now?
4. What gates must pass next?
5. What claims remain sealed?

This document answers a different question: **how did we get here, and which
roads have already been falsified?** Keeping those roles separate prevents the
roadmap from becoming an ever-growing lab notebook.

## 2026-09-06 archived roadmap experiment chronology

Archived when the user redirected development toward a broader persistent
language-and-action workload. The following retains historical selections
and measured results verbatim; its next-step statements are superseded by
the current roadmap. Frozen experiments retain their original contracts.

Source roadmap SHA256 before this move: `e2c79446f93a48160b30bebe70c75e77d6e3f37817988c5322705ef7cb27d9a6`.

Current endpoint: the [completed byte-coincidence comparison](../../experiments/r4_integrated/neuronal_splice_online_v3_result.md)
solves 4/32 fresh tasks versus 5/32 for its matched graded competitive reference.
Both retain 14/16 actions and all twenty old reading answers. The strict criterion
fails. Full run and independent audit total 157.90 seconds at 1,047.44 MiB observed
owned peak, after completed admission at 673.20 projected seconds and 1,300.06 MiB.
The [v3 result bank](../../benchmarks/neuronal_splice_online_v3.decision.json)
preserves all fixed exposures, controls and failed criteria.

The [saved learning review](neuronal-splice-online-v3-learning-review.md) finds
correct candidate payloads and EOS on all 28 fresh edits, versus 24/28 for the
reference; all 27 nonempty intended starts are uniquely ranked first and selected.
However, the candidate chooses STOP on 23 required edits and SPLICE on all four
KEEP tasks. One further edit has an incorrect span. Thus useful payload transfer
does not establish complete action superiority. Preserve this distinction and
close the cohort without additional rounds, endpoint replay or seed/rate sweeps.
The completed [operation review](neuronal-operation-intent-review-v1.md) selects
one generic beginning-aligned whole-request rail branch into those same locally
learned operation neurons. Preserve the old contextual branch and its reduction
order, with zero-initialized added synapses and separate activity normalization.
The existing operation head sees request activity only through selected
context/path correlations; direct request access is the selected hypothesis,
not a proven explanation. Implement this versioned source while retaining the
reader, span heads and actual local action/teacher loop. Reviewed ordinary QA
and a fresh comparison/cost freeze must precede any new acquisition. No host
verb parser, operation oracle, backward error transport or standalone fit follows.
The [resource review](neuronal-splice-online-v3-cost-review.md) keeps efficiency
part of that same integrated workload. Frozen comparison contracts stay unchanged.

The preceding [completed competitive neuronal splice learner](../../experiments/r4_integrated/neuronal_splice_online_v2_result.md)
solves 20/32 fresh tasks (19 real edits and one explicit STOP), versus 7/32
for its freshly trained marginal reference and 1/32 for each of three matched
controls. It retains 14/16 actions and answers all twenty old reader questions
correctly before and after acquisition. Neurons select the file, raw span,
payload and operation; the same persistent system learns locally after actual
execution without backpropagation. The strict all-fresh/all-retention criterion
still fails. Full run and independent audit total 135.77 seconds with
1,522.54 MiB observed owned peak, after completed admission at 583.71 projected
seconds and 1,879.49 MiB. The [result bank](../../benchmarks/neuronal_splice_online_v2.decision.json)
preserves the fixed source, complete scope, controls and evidence hashes.

The [five-arm comparison](neuronal-address-competition-comparison-v1.md) is now
closed. Initial neuronal address competition passes its mechanism-superiority
component: learned fresh edits exceed all controls and the matched marginal
reference. Document-order transfer is 6/8, continuation 8/8, value-role transfer
4/8 and operation/layout 2/8. The delayed value pairs pass; the five-document
interference pairs do not. These narrow literal-splice tasks do not establish
general software engineering, language understanding or the integrated gate.

The completed saved-only [learning review](neuronal-splice-online-v2-learning-review.md)
finds wrong initial address ordering in the payload failures; separate location
and STOP failures remain. The [resource review](neuronal-splice-online-v2-cost-review.md)
supports retaining the qualified executor and streamed receipts. Its selected
successor was [generic neuronal byte coincidence](neuronal-byte-coincidence-review-v1.md):
fixed dual-rail AND/match cells before the existing locally learned temporal
address synapses, with Boolean memory keys. It retains neuronal commitment, codec,
action heads and the exact same-instance direct local teacher. No error propagates
backward through the fixed cells. All six ordinary QA nodes and independent
saved review pass: arithmetic, typed storage, controls and actual execution before
correction, at 11.08 outer seconds and 278.19 MiB observed peak. The
[fresh five-arm comparison](neuronal-byte-coincidence-comparison-v1.md) subsequently
completed as v3 above, with the graded competitive reader as its matched reference.
The QA pass alone does not establish acquired skill. The v2 cohort remains closed, exposed
development evidence: no additional rounds, endpoint replay or rate/seed sweep.
V3 used newly withheld cases, an explicit transferred parameter prior with changed
feature meaning, and fresh complete cost admission. Neither completed comparison
advances the general benchmark gates.

The preceding [splice-v1 result](../../experiments/r4_integrated/neuronal_splice_online_v1_result.md)
and its [saved learning review](neuronal-splice-online-v1-learning-review.md)
remain historical evidence. Its diffuse off-target EOS behavior motivated the
current fixed threshold competition, but its different cohort is not a matched
baseline for subtracting v2 scores. Component, receipt, worker, data and synthetic
cost QA remain banked; their passed nodes are not repeated.

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
keeps the learning workload unchanged. Its completed cost admits 586.173 projected
seconds; the full run and audit finish in 248.580 seconds. The scoped 8/4/4/2
criterion passes with all controls at zero. Close this fit and pursue the active
neuronal action-interface review above; the strict 20/20 gate remains unmet.
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


## 2026-09-07 broader persistent language/action return

The prepared byte-coincidence comparison was completed and banked before the
broader return. The user stopped the automatic operation micro-gate direction;
its unexecuted source remains paused. The main roadmap was shortened and its
experimental diary moved here.

At source `032d908`, a 56-episode public stream ran on the same persistent
neuronal models with a shared 453-slot memory and one-pass local corrections.
Online and matched learning-disabled routes scored 0/24 acquisition first
responses, 0/16 fresh transfer, 0/8 delayed retention and 0/8 recovery. The simple
reference scored 12/24, 9/16, 8/8 and 0/8. Four long-request challenges remained
visible; within common input bounds the reference scored 6/12 fresh versus zero.
The run and independent saved review passed first launch, totaling 19.662127
seconds at 364.921875 MiB observed peak. Both final neuronal states are saved.

The [result](../../experiments/r4_integrated/persistent_language_action_v1_result.md)
and [bank](../../benchmarks/persistent_language_action_v1.result.json) retain
all exposure and evidence identities. Saved reviews find broader address,
answer-boundary and edit-location failures, rather than the old false-STOP issue.
The next shared source/design priority is request-conditioned answer construction
and stopping, preserving the broader workload, local learning, matched references
and honest recovery limits. No toy-perfect prerequisite or automatic extra
training rounds were selected.

## 2026-09-07 request-conditioned broader continuation

Source `5d9c302` continues the first session's actual online model, experience46
and whole453-slot tape. It adds request/prefix-conditioned all-address retrieval
at every byte, fixed mixed threshold inputs and direct local normalized
output-current learning. The old codec is held as a numerical prior; its old
successor traversal is not preserved. Two ordinary mechanics tests pass before
the complete broader continuation, without a fitted skill threshold.

The 56-episode continuation adds two notes and reaches505slots. Online and the
matched current-learning-disabled fork score0/24 acquisition first responses,
0/16 new request/structure transfer,0/8 earlier-session retention and0/8 recovery.
The reference restores its actual earlier history and scores13/24,11/16,8/8,0/8.
Each neuronal route receives24corrections,98byte/EOS offers and36actionlabels.
Across allroutes216responses and144actual executions complete; run and saved
review pass firstlaunch in117.971022seconds at406.921875MiB observedpeak.

The [result](../../experiments/r4_integrated/persistent_language_action_v2_result.md)
and [bank](../../benchmarks/persistent_language_action_v2.result.json) preserve
all measurements and continuation states. The [learning review](persistent-language-action-v2-learning-review.md)
finds identical reading outputs between neuronal routes, mostly repeated bytes
to the output limit. Every actual online splice lacks EOS and is rejected.
Local B/C/U publication and occasional address switching do not establish
effective sequence learning. The next shared question is alignment of local
sequence teaching with actual free generation, alongside measured simulation
cost; no gain sweep, extra fitting rounds or isolated operation gate is selected.

## 2026-09-07 actual-response output correction in the broader session

Source `fc7f614` continues the full v2 online numerical model, 505-slot tape and
experience count 92. One direct output-current correction binds to the first
actual response error; all-target B/C teaching remains separately declared.
An exact incremental prefix cache avoids 90.0007% of pair reevaluations on the
recorded trajectories. Ordinary arithmetic, atomicity and real execution
ownership checks pass without a task-performance prerequisite.

The new 56-episode public session interleaves related instruction forms and
introduces negation, reordered composition, arithmetic and new code structures.
Both neuronal routes score 0/24 acquisition first responses, 0/16 fresh transfer,
0/8 delayed retention and 0/8 recovery. The reference scores 12/24, 6/16, 8/8 and
0/8; supported fresh scores are 0/12, 0/12 and 5/12. Each neuronal route receives
172 target symbols and 36 action labels; online publishes 24 actual-event U
clamps and 24 B/C changes. Fewer output updates do not reduce supplied teacher
information. Memory stays at 505 slots, extending age without new capacity.

The [result](../../experiments/r4_integrated/persistent_language_action_v3_result.md)
and [bank](../../benchmarks/persistent_language_action_v3.result.json) preserve
216 responses, 144 actual executions and both final continuation states. Full
run and saved audit pass first launch in 60.5972151 seconds at 512.5234375 MiB
observed peak. Workload, teaching and receipts differ from v2, so the elapsed
ratio is not an isolated cache-speed comparison.

Saved features show broad overlap between correction contexts, while 22 of 24
output clamps target the first event and two the second; none teaches EOS.
Actual outputs change but do not solve the tasks. The next bounded architectural
review concerns context-specific local learning and effective use of already
offered sequence information within the same broader learner. It is not an
automatic extra pass on this now-exposed stream.

## 2026-09-07 context assemblies and broader actual projects

Source `07ab104` continues the full v3 online numerical reader, embedded old
 action values, experience 138 and whole 505-slot tape. The same reader/actor
adds target-independent local context recruitment and all offered sequence
updates, with inherited B/C/U/codec held. Inputs expand to whole 256-byte
instructions, four 512-byte files, 128-byte outputs and actual diagnostic bytes.
Six ordinary QA nodes and one affected repeat pass; no toy skill gate precedes
the broader return.

The frozen 56-episode session includes 48 new multiclause instructions,
missing program logic, three actual project chains and eight exact regressions.
Only the first twelve acquisition episodes receive corrections, offering 235
reader symbols and eighteen action labels per neuronal route. No new documents
are ingested. Across all routes, 216 responses and 144 actual executions complete.
Both neuronal routes score 0/24 acquisition, 0/16 fresh transfer, 0/8 retention
and 0/8 recovery; the simple reference scores 0/24, 0/16, 8/8 and 0/8.

The [result](../../experiments/r4_integrated/persistent_language_action_v4_result.md)
and [bank](../../benchmarks/persistent_language_action_v4.result.json) preserve
both final states and the passing saved audit. Full run plus review takes
560.1913689 seconds at 970.2421875 MiB observed peak. Online recruits 235 contexts
and makes 2,079 free output decisions versus 8,405 disabled. Whole sequence
storage and stopping change, but correct context transfer, composition and action
remain unmet. The saved reviews inform one shared successor; no extra fitting
round on this exposed stream or automatic byte/operation gate is selected.

## 2026-09-07 factorized routing and growing persistent experience

Source `c2b3f92` continues the actual v4 online state into two independent
learning/acting copies. The new shared local router compares fixed byte/pair/
relative-position field activity over every stored event owner, with float64
competition and direct own-emission outcome clamps. Separate target-free exact
preparation recruits the assembly whose outgoing byte receives its local teacher.
The old numerical priors remain intact. Three ordinary integration QA nodes and
one reporting QA node pass; there is no scientific byte/operation gate.

The64-episode stream gives eight corrections, changed forms, delayed exact
repeats, sixteen fresh combinations, eight three-attempt recoveries and sixteen
older regressions. Six documents add353 raw bytes plus6 EOS, growing complete
memory505→864 slots. Corrections offer163 external symbols per neuronal route,
with two internal uses and twelve action-head labels. Across three routes the
run completes240 attempts,156 actual child verifications,24 correction occasions
and18 document appends.

Online scores0/8 corrected first responses,0/8 changed forms,1/8 exact delayed
repeats,0/16 fresh,0/16 prior regressions and0/8 recovery. Disabled scores0/8 exact
repeats and4/16 prior regressions; the simple reference scores4/8 and10/16. Both
also fail every fresh/recovery task. All80 observations per route are supported.
The neuronal historical failures remain failures rather than forgotten skills;
the current online/control contrast nevertheless exposes adverse correction
effects. Online's one delayed success is the reading answer `idle`.

The [result](../../experiments/r4_integrated/persistent_language_action_v5_result.md)
and [bank](../../benchmarks/persistent_language_action_v5.result.json) preserve
both final states under checkpoint `a141230e...`,100,830,448 tensor bytes. Online
finishes with398 event owners and disabled235. The first full run and saved audit
pass in558.5371355 plus8.3754013 seconds,566.9125368 total, at723.078125 MiB
observed peak. Ordinary authored-data validation has one retained36-check harness
failure and a36/36 qualified-executor pass on the same once-built manifest,
total72 author-only children and no learner calls.

Saved learning, evaluation and cost reviews guide the shared continuation.
This exposed stream is not replayed or used as a toy-perfection prerequisite;
coherent wrong answers, shared routing interference and actual feedback use
remain central capability questions. Owned-response routing validation is the
selected immediate integral optimization, not an isolated speed/capability gate.

## 2026-09-07 local recognition and capacity integration

Source `ca72658b935050d1d9d913570e5985c1a1812580` adds one locally updated
recognition gain per input-defined event, retaining the old shared theta as a
frozen numerical prior. Input-only preparation precedes each current teacher
byte; the same reader supplies language responses and executed action payloads.
Explicit successors expand event capacity512→1024 and factual capacity1024→2048
without dropping old coordinates or fitting gains from prior lessons.

Four [ordinary affected QA nodes](input-defined-recognition-qa-v1.md) pass on
their first launch in19.2369283 seconds at880.23046875MiB observed peak. They
check local arithmetic, prior-score parity at zero gains, disabled and atomic
updates, actual action custody and crossing both old capacity boundaries.
They establish no new language/SWE result and load no experimental checkpoint.
The next workload is a72-episode broader continuation with longer retention,
new constructions/combinations and fewer corrections; its authoring and saved
audit are prepared separately from this completed implementation verification.

## 2026-09-07 sixth broader stream and resource cutoff

Released source `46839d7fcea50dc6e805db910a36616b0bd51589` runs one72-episode,
96-attempt-per-route stream from the complete V5 online checkpoint. Its once-built
manifest and42 passing author-only program witnesses precede the launch. Six
corrections offer135 symbols and nine action labels per neural route; twelve
complete documents grow factual memory864→1760. Input-only private recognition
adds135 online owners, reaching533, while disabled retains398.

The watchdog stops the run after840.4533474 seconds at890.42578125MiB observed
peak. All first60 episodes finish for all three routes, then two online C00-E
recovery attempts finish. The182 durable attempts,18 teachers,36 appends and92
actual execution receipts are preserved. Online scores1/6 exact delayed,0/24
fresh and0/18 old regressions; disabled scores0/6,0/24,1/18 and the reference
4/6,0/24,11/18. Both observed recovery attempts fail, but the full recovery
outcome is unavailable. No final checkpoint or completed-run audit is produced.

Three bounded saved-only partial analyses check recorded local algebra, execution
and cost. The resource extractor's inherited8-MiB row guard fails once; the
corrected reader passes on fresh output, with both receipts retained. No model
or teacher replay repairs the failed session. The [result](../../experiments/r4_integrated/persistent_language_action_v6_result.md)
and [bank](../../benchmarks/persistent_language_action_v6.result.json) own exact
evidence and limitations. V5 remains the last recoverable complete state.

Positive recognition learning changes routing but causes broad new-owner
attraction: the first new reading owner begins60/62 online responses. The
recorded workload performs3,357,755,083 factual pair comparisons; root selects
an exact all-address fixed-neuron tournament as an integral optimization, with
safe-boundary persistence needed for future sessions. A broader learning-design
review addresses relevance and composition rather than starting another
automatic correction tweak, exposed replay or toy-perfection gate.

## 2026-09-07 exact factual competition integration

Source `af975cd9c3ed6fda8a11b8c47c10df4ad9090625` integrates an all-address
fixed comparator/copy-relay tournament into the persistent recognition actor.
Original factual scores and downstream numerical state remain unchanged; new
metadata and work counters describe the different forward circuit. The two
[ordinary QA nodes](factual-address-tournament-qa-v1.md) pass first launch in
13.9122443 seconds at688.4765625MiB. Eighteen valid old/new competitions and
eight input rejections cover exact winners, ties, levels and buffers; two fresh
actors make four responses, two actual STOP executions and eight local teacher
offers with exact action/update parity. One in-memory strict state load passes.
No experimental checkpoint or V6 task is replayed. This is an integral
simulation improvement, with broader runtime benefit still unmeasured.

## 2026-09-07 safe-boundary persistent state

Source `cf26310d4ee96647db6eac4207f0f0a0161f66c5` adds durable initialization and
transaction intents, immutable route payloads and one atomic HEAD. Ten ordinary
interruption fixtures pass first launch. The actual actor continuation test
reaches restoration but incorrectly demands distinct empty Python cache tuples;
test repair `8338af3938fb45c528b66529e1bb24d8b27cf72a` checks actual tensor storage
and the affected node passes. [QA evidence](persistent-session-checkpoint-qa-v1.md)
retains both launches:37.8355623seconds total and1232.08203125MiB maximum observed
peak, eight fresh actors, eight teacher symbols and four actual children across
failed and repaired fixtures. This is mechanics verification, not a broader
session result. No V6 state is reconstructed or old workload repeated. Paired
dendritic source/request learning and fresh broader workload preparation proceed
as the shared next integration.

## 2026-09-07 paired dendritic learning integration

Source `cc4275969953e4f19da82d708bc3b926fb1da673` adds local paired request/prefix
contacts to the same acting reader. The two [ordinary QA nodes](dendritic-request-binding-qa-v1.md)
pass first launch in13.8375434seconds at763.66015625MiB: one isolated bank and
two combined actors,22branch evaluations, six source and ten correction-symbol
offers including declared disabled and rejected proposals, one actual child.
The old reader priors remain fixed. The V7 continuation adapter preserves the
complete V5 state and introduces empty contacts; no historical checkpoint was
loaded in this QA. A48-episode broader stream and safe-boundary runner follow
without another toy acquisition gate. Semantic transfer remains unmeasured.

## 2026-09-07 V7 author preparation

Source `13043ed670465856bbdbd37930636d4f9e3d5938` adds the durable broader runner
and releases one pure workload build. [Author/metric QA](persistent-language-action-v7-author-qa.md)
passes first launch: one synthetic outcome test, one48-episode build and all28
nominal edit witnesses. Zero learners or experimental checkpoint loads occur.
The manifest binds actual prior regression observations and separately declares
750 source symbols plus71 answer symbols per neural route. New requirements,
changed instruction forms, actual project chains and fixed teacher-free recovery
share one persistent session. Root selects a1800+120-second/2048-MiB complete
envelope; learner utility is not established by these author checks.

## 2026-09-07 seventh broader stream with durable partial state

Released source `a1d900a688d353fcc00e43b86ccb14b1547f3065` executes the frozen
48-episode stream. The1800-second runner ceiling and30-second admission reserve
stop it cleanly after185/192 attempts, at1771.0604454seconds and818.6484375MiB
observed peak. All fresh, prior and fixed recovery cases finish; seven late
acquisition outcomes remain unknown. Online scores0/4 changed forms,1/2 observed
exact repeats,0/16 fresh,0/12 prior and0/8 recovery. Disabled retains the one
previously correct V5 reading, while online loses it; the reference retains
eight prior successes. No correct neuronal software edit occurs.

Four answer corrections offer71 symbols plus six action labels per neural route.
Twelve whole sources offer750 additional causal next-symbol observations and
grow factual memory864→1614. The same online reader publishes local branch
learning; the matched control computes proposals without publishing them, with
factual writes in both. The128 actual child receipts,36 appends and12 teacher
rows remain bound to222 safe checkpoints. HEAD has no outstanding intent or
tail. Numeric payloads are authenticated, but no actual V7 restore is performed.

Saved review source `fe627208900e27253eb2ef708ad0225df4b0d75c` fails because its
initial-binding loop shadows the previous boundary hash. The narrow repair in
`827a2de427f6b69c87d61b2f7069835c32da5b3e` passes on unchanged saved evidence.
The failed review's12.9411646seconds, corrected review's42.3644378seconds and
banker's2.7871129seconds are retained: all four outer processes total
1829.1531607seconds. The [result](../../experiments/r4_integrated/persistent_language_action_v7_result.md),
[bank](../../benchmarks/persistent_language_action_v7.result.json) and
[outer receipt](../../benchmarks/persistent_language_action_v7.measurement.json)
separate incomplete scientific outcomes from completed saved checking.

The [cost review](persistent-language-action-v7-cost-review.md) measures
917.9653919seconds of checkpoint work after subtracting attempt/begin overlap.
Root selects immutable-payload ownership design to reduce repeated integrity
reads, preserving all mutable-state and restore checks. The parallel
[learning review](persistent-language-action-v7-learning-review.md) identifies
public-observation relevance as a bounded design question, not a measured
causal diagnosis. These feed one broader continuation; no exposed replay,
gain sweep or automatic toy gate follows.

## 2026-09-07 owned checkpoint payload integration

Source `768e15c4a5a460c9bc57b15842f8dc1110d53edf` adds authenticated Windows
NTFS read ownership of newly published immutable route payloads. Continuous
noninherited handles deny write/delete sharing; later boundaries check file
identity and path without streaming the same protected bytes again. Fresh opens
and restore retain full hashing, as do unsupported-platform boundaries. Mutable
actor/tape/source/journal checks and atomic publication ordering remain intact.

The [three ordinary checks](persistent-checkpoint-owned-payload-qa-v1.md) pass
first launch in18.9013638outer seconds at1238.98046875MiB observed peak. Thirteen
byte-file refusal cases and ten opaque interruption fixtures accompany one
actual actor continuation fixture: four actors, six free calls, two STOP
children, four teacher symbols, six action labels, three publications and one
restore of two locally created tensor payloads. One injected path failure
before new snapshot leaves HEAD and intent intact, then the fixture's original
commit succeeds. No experimental checkpoint, V7 lesson or missing task is replayed.
These checks verify reduced repeated byte work under ownership, not a measured
broader-workload speedup or an additional capability gate.

## 2026-09-07 reusable context/content learning selection

The V7 learning review identifies a relevance limitation but does not diagnose
the cause of zero transfer. A proposed field-patch overlay is superseded because
more exact matching alone supplies neither learned operation reuse nor variable
binding. The [selected design](public-observation-sequence-context-v1.md) uses
32 stochastic recurrent context cells, two content cursors and fixed neuronal
bind/advance/read relays in the same acting reader. Direct own-synapse eligibility
receives declared observed-symbol and coarse public-consequence signals, with
weights held through a common input scan and actual/correction branches.

Source/math review corrects future-request visibility, all-field cursor masking,
RNG reuse across teaching, teacher-prefix prior accounting and the extra
returncode-EOS address. Both future neural routes will independently continue
V7 online; its numerical state is retained, but the new stochastic policy does
not promise old behavior parity. The design is selected for source implementation,
not a completed model or evidence of useful generalization. Larger raw-experience
and retained-memory budgets remain to be priced before a broader release.

## 2026-09-07 recurrent content implementation and ordinary QA

Source `bf9a9f0451b3d7de1d17f9e9782f872dc69f112e` implements the selected
controller, immutable causal frames and same-reader/action integration. The
complete V7 reader remains a numerical prior; new stochastic context/gate/mode
and literal cells learn through held local eligibility. Transient observations
preserve the original tape object; explicit admitted sources retain existing
factual writes. Disabled resolution advances its own RNG while holding learned
weights and baselines. Source review closes tensor-bearing trace serialization
and controller-only idle guards before execution.

The [two ordinary checks](recurrent-content-qa-v1.md) pass first launch in
18.527647 outer seconds at866.30859375 MiB observed process-tree peak. They
contain124 predictive offers, including two in an intentionally rejected
transaction,390 categorical draws, seven held-prior calls and one actual STOP
child. Separate raw-frame relays receive47 symbols without model predictions.
Scalar arithmetic, causal visibility, teacher branching, disabled acknowledgement,
late rejection and strict local restoration pass. The
[cost review](recurrent-content-cost-review-v1.md) binds the saved evidence and
separates sparse current work from dense transaction validation and custody.

These fixtures do not establish language transfer or useful recovery. Numerical
V7 adaptation remains source-only and no historical checkpoint is decoded.
The next shared work is authenticated initialization, explicit transient-source
persistence and a priced broader stream with public checks; no gain sweep,
automatic missing-case replay or toy-perfection prerequisite follows.
