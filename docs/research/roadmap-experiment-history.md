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

## 12. Retired directions

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

## 13. Why this history is separate

The main roadmap should answer:

1. What are we trying to build?
2. What design principles constrain it?
3. Where are we now?
4. What gates must pass next?
5. What claims remain sealed?

This document answers a different question: **how did we get here, and which
roads have already been falsified?** Keeping those roles separate prevents the
roadmap from becoming an ever-growing lab notebook.
