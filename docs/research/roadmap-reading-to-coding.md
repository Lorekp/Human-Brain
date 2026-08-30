# Roadmap: from reading to coding

Status: living document, updated per experiment wave. Owner: orchestrator.
North star (user): real-world use, primarily CODING. Intermediate requirement:
a network that can read, remember, and answer about text (Bible first) —
"a network that can't remember the Bible won't remember code."

## Where we are (evidence-anchored)

- Million-scale demonstrated: 1M-cell CUDA text training ran end-to-end
  (22.7 tok/s grow / 22.1 read, VRAM 2.7/12 GB; full KJV ≈ 15 h on this box).
  Scale ladder measured end to end: CUDA ≤200k any activity; CPU frontier
  200k–6M; procedural sparse owns ≥1e7 configured (only architecture that
  fits); tiering ceiling ~1.9e8 residents on this box.
- Root cause of no-learning found and measured: readout feature starvation —
  every prompt collapses to ONE point in eligibility space (cosine 0.9989),
  so the readout emits the unigram prior (`variable-0` everywhere). Race is
  argmax-current; it decides correctly on whatever features exist. The
  features are the problem.
- Falsified axes (banked, never revisit under ridge + current features):
  presentation schedules, codebook geometry, replay prioritization,
  EI-fraction, local-rule constants at current feature scale.
- QA-v1 exists: language-side gate implemented (fixture corpus, span
  accuracy, degeneracy metrics first-class), Bible corpus designed (PG10 =
  KJV, 66-book spans, 1.40M tokens, synoptic-leakage-safe split).

## The one critical insight

Reading and coding fail for the SAME reason: both require the readout to
distinguish contexts that current features cannot separate. A network that
cannot tell two verses apart cannot tell two function signatures apart.
So the roadmap to coding runs THROUGH reading — exactly as the user
intuited. The pivot is not a detour; it is the mainline.

## The 11 properties as the organizing principle (user directive)

The user's property table (recurrence, integration, global workspace,
persistent state, metacognition, attention model, self-model,
autobiographical memory, agency, embodiment/closed-loop, valence) is not a
wish list appended to the roadmap — it IS the roadmap's spine. Reasoning
recorded here so every phase decision references it:

**1. Why the current architecture scores ABSENT on most of them — one
mechanism.** The feature-starvation finding (cosine 0.9989) explains more
than the transfer failure: a readout that sees ONE point in state space
has nothing to loop back (no usable recurrence signal), nothing to
integrate (states don't differ), no workspace content worth broadcasting
(one candidate = no competition), and no internal state that persists
meaningfully between inputs (it's identical between inputs). Fixing R1
(features) is simultaneously the first step toward properties 1/2/4.
The EI sweep proved state diversity is purchasable but currently mispriced
(separation ×100 only where capacity collapses) — multi-timescale traces
are the priced version of the same purchase.

**2. Property → phase mapping (with honest dependency order):**
- Strong recurrence (P1): partially exists (fanout carries prior-token
  state); becomes USEFUL at R1 when traces differentiate contexts; fully
  realized when recurrent loops re-enter computation (R3+).
- Integrated representations (P2): requires P1's differentiated states to
  bind across populations — R1/R2 deliver the substrate; concept-chunk
  tokens (representation study's named fallback) are R2/R3's contribution.
- Global workspace (P3): the first-spike race IS a bottleneck competition;
  missing half is BROADCAST — winner re-enters the network as input. That
  is a small, concrete mechanism (winner-feedback loop) scheduled as an
  R3 experiment, gated on R1 (broadcasting a constant is pointless).
- Persistent internal state (P4): eligibility traces already provide the
  physics; R1 makes it informative; R3's episodic-store decision completes it.
- Metacognition (P5): margins/confidence exist but never re-enter
  computation (scored PARTIAL-as-seed). R3 attaches them to answers
  (calibrated abstention); R4 uses them for self-checking before submission.
- Attention model (P6): emerges from workspace + QA framing (which context
  segment wins determines the answer); measurable via perturbation tests in R3.
- Self-model (P7): honestly NOT EXERCISED by token prediction alone;
  earliest credible seed is R4 (the system representing its own pass/fail
  history on coding tasks — autobiographical by construction).
- Long-term autobiographical memory (P8): R3 directly; the episodic-store
  question is its central design decision.
- Goal-directed agency (P9): enters at R4-R5 (mini-swe agent loop: act,
  run tests, observe failure, revise). Before that, honestly absent.
- Embodiment / closed loop (P10): R4's sandbox IS the closed loop — code
  actions produce test results that alter future observations. First true
  closure point of the whole program.
- Homeostatic drives / valence (P11): seeded by turnover/activity-EMA
  homeostat (exists dormant) + margin-valence mapping; scheduled R3+ as
  stabilization, not motivation-theater.

**3. The sequencing logic the properties impose.** The properties form a
dependency DAG, and it matches the phase order: P4/P1 (state, recurrence)
→ P2 (integration) → P3/P6 (workspace, attention) → P5 (metacognition)
→ P8 (autobiography) → P9/P10 (agency, closure) with P11 as continuous
stabilizer. Trying to install later properties without earlier ones is
how you get "an enormous, rich, chaotic, high-dimensional system that
isn't doing anything particularly useful" — the user's phrase. Each phase
adds organization to complexity ALREADY distinguishable from the previous
phase's vantage point.

**4. Distributed representation (user's second directive) inside this
spine.** Explicit token identities were falsified as insufficient for
transfer (codebook experiment), but random-distribution was equally null:
distribution must be STRUCTURED (similarity-projected or concept-mined).
The representation study's ranked recommendation stands, now gated behind
R1: distributed codes organize the input side, multi-timescale traces
organize the temporal side, and the two together are what give the
readout something to integrate (P2) and the workspace something to
compete over (P3). Representation choice is therefore not one decision
but a standing constraint on every phase: no component may reintroduce
grandmother-cell identities — assemblies only.

## Roadmap

### Phase R0 — Multi-timescale features — EXECUTED, FALSIFIED (2026-08-23)

Outcome (preregistered gates, one-shot): eligibility traces at 8 taus
(16→800-token windows) reached only 6.8e-05 separation vs required >0.10.
The R0-b successor families (spike-count windows, voltage quantiles,
assembly-overlap) also failed G1 (~1e-4; assembly+random passed at 0.102
but was capacity-regressive). Full verdicts in
experiments/r0_multitrace/gate_results.json + experiments/r0b_state_features/.
Infrastructure built here survives: v6 checkpoints, K-bank native mirrors
(Rust AVX/parallel/frontier + CUDA), scipy-class solver, per-family gate
runner. FALSIFIER CLAUSE FIRED → next candidates evaluated by strategy
report (in flight): race/output winner-feedback loop, associative-memory
hybrid, learned output map over cell dynamics. The pivot is NOT a failure
of the roadmap — R0 was designed to test the cheapest feature family first;
it cost days, not months, and named its successors in advance.

### Phase R0-b/race — Decision layer — EXECUTED (2026-08-24)

Race-pivot implemented per protocol (broadcast engine P-S/P-R, margin
consumers; 21/21 tests green). N=96 fixture gates: TERMINAL INELIGIBLE —
every arm exact=0 (G-P2 fail) though broadcast de-collapsed answers
(6 distinct, modal 0.167, replay 42/42). N=1024 not spent per
stop-on-failure. Verdict: docs/research/race-pivot-gate-verdict.md.

### Phase R0-d — Custom architecture pivot — EXECUTED; SUCCESSORS ACTIVE

Reservoir-computing family falsified for QA reading after the full chain:
features (R0/R0-b), associative store on reservoir keys (R0-c), scale alone
(R2 pilot 1.19% at 131k), decision layer (race-pivot). Custom-architecture
research (custom-architecture-directions.md) ranked five directions; two
executed week-1:

- **U-STATE (SDR union memory): NEGATIVE_WITH_LOCALIZATION_V2**
  (40a2934). Separation PASSED (concept unions 0.402–0.607 vs bytes
  0.166–0.190; collapse tail 290× better) but decode failed everywhere
  (best 2.19% < unigram 4.54% < bar 6.19%), flat across the λ sweep.
  P3: concept self-store recovers speakers at 38.3% in TRAIN books but
  0% on VAL — capacity intact, generalization zero.
- **APICAL-AND (two-compartment coincidence encoder)**: G-A1 conjunction
  index failed in all 18 arms (<1.5×); decode analysis pending.

**The cross-cutting finding (design constraint for everything downstream):
nine preregistered executions show separation and storage capacity SUCCEED
while generalization of stored context to unseen question–context pairings
FAILS.** The deficit is not representational, not capacity, not readout —
it is the absence of any mechanism binding paraphrased-question space to
passage-content space. Architectures that only reorganize HOW context is
encoded cannot fix this; what is missing is a trained mapping between two
different content spaces (question-space and passage-space), which in
brains is exactly what learned long-range connectivity provides and what
every fixed-connectivity variant tested here lacks by construction.

### Phase R2 — Reading at scale (the demonstration, weeks)

Bible-QA v1 runs at 128k–1M CUDA cells: train on 49 books, hold out the
17-book block (minor prophets + gospels/Acts), measure answer-span accuracy
and free-run exact match vs the n-gram baseline. Gate: beat the unigram and
trigram baselines on held-out books at matched budget. Scale path already
measured: 1M now; procedural adapter (3 layers specified) carries text
training to ≥1e7 configured where CUDA cannot fit.

### Phase R3 — Memory that persists (autobiographical seed)

Long-context recall tasks over the trained net: "who said X" (chapter-
distance attribution), entity-tracking across books, verse continuation
from arbitrary offsets. These exercise recurrence/integration/persistent
state (properties 1/2/4) and proto-metacognition (5). Failure modes here
directly inform whether weights alone suffice or an explicit episodic
store (sparse associative memory over assemblies) is required — that
decision gates Phase R4 design.

### Phase R4 — Code as a second language (the goal)

Reuse the identical stack with the simple-SWE corpus (already built:
typed callable families, hidden-case oracle grading, sandbox). The
transfer question — does Bible-training help code-learning — becomes a
measurable two-corpus experiment (pretrain on KJV, fine-tune on code vs
code-only). Coding gates: pass@1 on held-out task families with hidden
tests, all-or-nothing per family, exactly like every gate in this repo.
Milestone ladder below Gemma already defined (M0 beat random completion
→ M1 beat n-gram → M2 beat template heuristic → M3 beat tiny neural LM).

**Public-TRAIN bridge status (2026-08-28).** The smallest closed coding loop
now works end to end on a scalar template: execute an identity source, reduce
the visible residual to observations, let fitted-cell races elect actions,
forward those actions into a separate fitted-cell role/place builder, generate
source directly, and run the result in the canonical sandbox. R3-PAC0 emitted
exact `n - 9`; R3-PAC1 retained the entrypoint across identity execution,
copied magnitude `2` from completed residual feedback, and emitted exact
`n - 2`. R3-PAC2 then added a second source shape: after real list execution it
physically emitted `MAP` and `ADD`, copied entrypoint, parameter, magnitude `3`,
and the repeated iterable, reached ordinary EOS, and emitted exact
`[item_0 + 3 for item_0 in values]`. Its scalar-shape and donor-payload
interventions emitted exact `values + 3` and map `+1` controls. All three actual
programs passed 36 TRAIN cases. PAC2 does not isolate observation-only shape
causality or cross-shape sign causality: the prompt identifies MAP, all MAP
donors are positive, feedback remains host-factorized, and both builder
skeletons are fixed. This does **not** open M0 or any protected split. PAC
probing now ends; the next step is one frozen generic cell checkpoint across a
fresh four-family self-posed coding cohort, with raw diagnostics, at most two
revision cycles, and cell-off/random controls. The closed typed-projection
tuning loop remains closed.

That cohort is now spent. All 12 cell completions were valid, sentinel-closed
Python, and raw feedback changed the selected program on three of four tasks,
so syntax validity and the closed-loop plumbing are no longer the immediate
bottleneck. Every completion nevertheless collapsed onto two whole programs
from the frozen corpus (seven fallback lookups and five inner slices). Final
hidden scores were 5/36, 0/36, 2/36, and 0/36, with 0/4 full task passes and a
terminal `STOP_FRESH_CODING_CHALLENGE`. This is a useful architectural
localization, not a reason to tune the spent cohort: the next increment must
replace whole-sequence continuation with a generic grammar/type stack whose
individual productions, operators, and arguments are selected by fitted-cell
races. Host code may enforce syntax and compile the selected tree, but it may
not supply task-specific skeletons. The next validation remains self-posed and
precanonical; M0 stays sealed.

A cheap prompt-last recurrent variant then reset and replayed the full natural
task plus generic typed-hole state before every production race. It still
missed production zero on both fresh compositions (fitted 0/2; untrained 0/2),
localizing the failure before recursion or compilation and ending that variant
without a parameter sweep. The active replacement is structurally different:
a persistent sparse prompt assembly plus generic hole-role cells drives a
local three-factor production-cell readout at every typed decision. Prompt and
role state are refreshed without replaying generated source through the same
recurrent bottleneck.

That replacement passed its physical mechanism smoke, then stopped honestly on
its first fresh binary-scalar composition. The fitted checkpoint recorded 51
local updates and 56 recruitments, but both the target prompt and a fixed prompt
swap elected root `variable-1`; the target program scored 1/8 cases. All races
were legal and competitive and inference left the checkpoint byte-identical,
so the failure is prompt-to-root discrimination rather than cell dynamics,
typed masking, or compilation. Exact localization showed the four teachers
themselves replayed only 1/4: shared signature/role cells caused competitive
last-label overwrite, and two reversed prompts were exactly aliased.

Ordered adjacent-token cells removed the exact alias but not the overwrite.
Conjoining those lexical cells with canonical signature + current typed-hole
role, while removing raw structural cells from plastic recruitment, then raised
both public teacher sets to exact 4/4 replay without changing cell count,
learning constants, or race dynamics. Its different fresh target still stopped
at root `variable-1` (1/8) rather than composing unseen `absolute → add`; the
fixed add-prompt control replayed its taught five-production tree exactly. This
is now a genuine unseen-context transfer failure. The next work is a visible,
freely rerunnable compositional development matrix plus root-current
localization, not another chain of frozen one-shot micro-probes.

That visible matrix closed the static whole-prompt decoder family. The retained
role-gated baseline replayed 4/6 teachers but composed 0/6. A disjoint additive
lexical/role bank and a four-branch lexical-by-role coincidence head each fell
to 0/6 teacher replay; a generic typed place key recovered only 1/6 and emitted
the same last-taught program for every case. Localization found that every hole
still received one 78–84-cell prompt union, while different prompts overlapped
by 28–60 cells. Role branches therefore collapsed before the physical race;
depth/child tensor factorization can repair unseen role conjunctions but cannot
recover lexical alignment already erased by the union. The failed uncommitted
heads were removed and the last working decoder restored. The immediate coding
slice now shifts to a generic typed edit loop whose cell-raced actions receive
real execution outcomes; role-conditioned prompt-span attention remains the
separate reading/language repair.

The first edit-loop slice then stopped without a retry. Seed 17 raced a generic
scalar bank (`absolute`/`add`/`subtract`, both operand orientations, and signed
single-digit literals) for eight public edit proposals from `variable-0`.
The exact proposals were `x-(-8)`, `abs(x)`, `x+0`, `x-8`, `abs(x)`, `x-4`,
`x-7`, and `abs(x)`; every proposal remained 0/4 on the public `abs(x)+3`
canary, so the host oracle undid all eight and the incumbent stayed unchanged.
This is `STOP_BLIND_TYPED_EDIT_SEARCH`: fitted cells causally selected every
edit, but the host stored novelty/policy currents, reduced outcomes to a score
delta, and accepted strict improvements. It is therefore a negative plumbing
probe, not cell-only feedback learning. The bank will not be narrowed and the
seed/budget will not be searched. The successor must put raw execution
observations into a persistent learned cell/synapse policy so proposals are
conditioned rather than brute-forced.

The first target-free sensorimotor successor stopped even earlier, at its
exploration seam. Raw before/after executions entered fitted thermometer and
opponent relation populations, and a sparse local readout was prepared to bind
each effect to the physical motor winner through an efference copy. Attempt 0
aborted before learning because the inherited 0.01 pA output heterogeneity put
all 20 inverse cells in one 0.05 ms spike bin. With no record, learned state, or
artifact produced, one target-independent repair reused the already-frozen
4 pA motor dispersion. The sole adjudicative run then completed one learning
episode (`OFFSET_ROOT[-8]`) and stopped at episode 2 when the persistent motor
population produced four simultaneous spikes. Coverage, checkpoint freeze, and
both ±3 gates remained unopened. This is `STOP_NON_UNIQUE_MOTOR_RACE`; the seed,
dispersion, reset policy, curriculum, and action bank will not be tuned. It
closes this homogeneous persistent-population babbling scheduler, while leaving
inverse effect recall itself untested. The next increment must obtain actions
from a structurally unique competition or from environment-triggered policy
updates, not another parameter repair of this scheduler.

The structurally different reward-modulated actor reached that next seam but
failed its public atomic readiness check, so composition remained unopened.
Each public goal executed all ten fixed typed actions in round-robin order; raw
L1 progress became a diffuse signed third factor on a sparse action row versus
a masked sink, while pacemaker bias was clamped out of inference. Three epochs
produced 300 balanced trials, 300 local updates, and 375 recruitments. The first
visible replay exposed the expected ±2 compromise under shared shaped reward;
one standard mean-centered actor-critic advantage repair removed all race ties
without changing seed, schedule, anatomy, or learning constants. Fresh atomic
replay still reached only 4/10 exact actions: `DONE`, `ADD2`, `ADD4`, and
`SUB3`. `ABS` routed to `ADD4`, while five other offsets routed to adjacent
magnitudes. This is `STOP_PUBLIC_ATOMIC_REPLAY`. G-RMA1 (`ABS → ADD3 → DONE`)
was never run, and the cumulative-feature actor receives no further reward,
race, epoch, or bank tuning. The surviving seam is reward-modulated local
action plasticity; the next representation increment must create exact signed
magnitude cells (or an equivalently margin-producing stage) before any
multi-edit target is opened.

That representation increment passed cleanly. A fixed, target- and
action-independent opponent projection transformed each cumulative signed
threshold bank into exact-magnitude cells; equality cells passed through
unchanged, and only the resulting spike mask reached a fresh fitted-cell
population and the same sparse local actor. After the same 300 balanced public
motor trials, the frozen seed-17 checkpoint selected all 10/10 atomic actions
exactly on a fresh argument grid. Every race was unique and inference preserved
the checkpoint hash. This is `PASS_EXACT_MAGNITUDE_ATOMIC_REPLAY`: it validates
the magnitude representation and licensed one composition trajectory, but it
does not itself establish composition or open M0, Simple-SWE, held-out, or C1.

The licensed G-RMA1 trajectory is now spent and stopped. From identity on the
public `abs(x)+3` target, the expected actions were `ABS → ADD3 → DONE`; the
frozen actor instead selected `ADD3 → ABS → ABS`, yielding `abs(abs(x+3))` and
outputs `[10, 3, 4, 13]` rather than `[16, 9, 4, 13]`. All selections were
unique and the checkpoint remained immutable, so this is not a recurrence,
race, or atomic-recognition failure. It is `STOP_G_RMA1`: the greedy
goal-conditioned policy lacks a representation of noncommutative edit order
and remaining horizon. There is no retry or tuning of this actor. The next
increment must represent multi-action consequences and prove itself on a
visible mixed-order matrix containing both outer-then-inner and
inner-then-outer programs before any fresh composition target is spent.

That multi-action increment now passes its full visible development matrix. A
target-free primitive corpus over values `[-20,20]` grew 410 local
pre-value × motor-efference × observed-post-value synapses in ten action banks;
there is no composite target or expected order in that learning stream. The
readonly successor then predicted two plies in fresh fitted-cell races.
Per-case exact-value coincidence spikes entered a materialized, hashed fixed
synaptic projection into 100 ordered-pair cells, and the winning pair spike
reached the ordinary ten-cell motor bank through another fixed projection.
Only that motor winner was applied to the real program; no candidate program
was host-executed or host-ranked.

On both visible grids, every combination of magnitudes 1/2/4, add/subtract,
and outer-versus-inner absolute placement passed: 24/24 exact action orders,
program tokens, and outputs. The run recorded 47,520 unique successor races,
7,200 cell-pair predictions, 72 selected real executions, and immutable
successor/value hashes. This is `PASS_HORIZON2_SUCCESSOR_VISIBLE`. It solves the
specific horizon/order defect with a learned cell world model, while remaining
a bounded exact-value scalar planner rather than broad coding evidence. The
withheld magnitude-3 paired transfer (`abs(x)-3` versus `abs(x-3)`) is absent
from this learner and now lives behind a separate persisted-PASS authenticator
and one-shot artifact guard.

That paired boundary is now spent at `STOP_HORIZON2_SUCCESSOR_FRESH_K3`.
The outer-subtract arm generated exact `ABS → SUB3 → DONE`. The inner-subtract
arm's first pair race also chose the correct `SUB3 → ABS`, and its selected
real action produced the expected intermediate values. Replanning then aborted
before its second action: exhaustive pair generation reached a candidate whose
intermediate `-21` lay outside the primitive learner's pre-value support
`[-20,20]`, so all 65 successor value cells received the same base current and
spiked together. The frozen checkpoint and recorded pair/scoring hashes stayed
unchanged. This is a useful boundary rather than a reason to retry k=3: the
world model needs a physical `UNKNOWN` successor state (and pair scoring that
can continue past unknown branches) before the same planning pattern is lifted
from scalar values into typed AST site/rewrite state.

That first typed-AST lift now passes its public visible matrix. A total
successor with an explicit physical `UNKNOWN` state kept unsupported branches
in the bank, while the readonly typed world model predicted every current
program site × nine generic wrap actions plus one global `DONE`. Fixed
projections drove separate candidate, site, and action races; only the winning
site/action actuator mutated the real program. All 8/8 one-edit Int tasks
selected the exact action and path across depths 0, 1, and 2, recording 1,776
successor races and 24 selection races with immutable checkpoints. This is
`PASS_AST_SITE_VISIBLE`, narrowly: H=1 public development with fused wrap
actions and a fixed six-case output goal, not generic AST synthesis or transfer.
The next increment factorizes site → schema → operator/production → operand
binding → literal and carries the total successor into bounded H=2 planning
over an Int/Bool/List development matrix.

The smallest genuine H=2 typed-AST slice now passes 3/3 visible Int tasks.
All incumbent sites entered the first-ply bank, and each first winner was scored
from a learned-cell second-ply root-wrap prediction; selected first edits alone
were actuated before real observation and replanning. This recovered both
noncommutative orders and a real path change from child `(0,)` to root `()`.
Exact enumeration evaluated 2,439 sequence cells across nine decisions and the
sole artifact run took about 114 s, so B=4 physical beam selection is the next
runtime optimization. `PASS_TYPED_AST_HORIZON2_INT` does not establish an
arbitrary hypothetical second site, Bool/List, fresh/sealed transfer, M0,
Simple-SWE, held-out, or C1 evidence.

The first B=4 runtime variant then stopped before second-ply expansion. It
scored 39 first candidates in 2.44 seconds, but fixed intermediate-value
currents gave nine simultaneous beam spikes on both one-site tasks and a
two-cell preliminary value-race tie on the site-change task. No candidate was
retained or expanded. This is `STOP_TYPED_AST_HORIZON2_INT_BEAM`, banked without
changing B, currents, weights, or ordering. It localizes the next speed layer to
a tie-safe physical candidate representation; it does not invalidate the
exhaustive H=2 pass or license a host top-k shortcut.

The tie-inclusive successor retained complete simultaneous-spike tiers and
recovered both root-order tasks. It kept 56 of 160 first candidates and scored
506 terminal sequences instead of 2,439 exhaustive, but stopped at 2/3 after
26.51 seconds. The child-to-root task ended on two genuinely different programs,
`add(abs(x), 3)` and `abs(add(x, 3))`, whose outputs each matched 4/6 cases and
therefore drove the same terminal current. This is
`STOP_TYPED_AST_HORIZON2_INT_TIER_BEAM`: the top-B mechanics are repaired, while
the remaining ambiguity belongs to output-only terminal representation. The
factorized planner must carry site/structure evidence into scoring; no fixed
index tie-break or beam-constant tuning follows.

The first uncertainty-driven extension then banked a narrower implementation
boundary. It cached 58 first-candidate expansions and evaluated 634 terminal
sequences across six physical races in 30.42 seconds. Crucially, expanding
successive complete tiers resolved the earlier child-to-root ambiguity and
selected the exact `(0,)::WRAP_ADD[1]` action with 6/6 coincidence. All three
trajectories nevertheless stopped later because the variant enumerated every
first-ply tier up front: after the physical STOP tier fired, unsupported
negative-current candidates below that frontier never spiked within the fixed
window. This is `STOP_TYPED_AST_HORIZON2_INT_ANYTIME`, banked without a retry.
The next named mechanism ends admissible enumeration at the STOP tier while
preserving complete simultaneous-spike tiers and uncertainty-driven expansion.

That mechanism now passes. `PASS_TYPED_AST_HORIZON2_INT_ANYTIME_V2` ends each
complete physical enumeration at the first tier containing STOP, admitted 198
of 252 first candidates across the nine decisions, and excluded 54 cells below
the physical frontier without awaiting them. Cached expansion evaluated 738
unique terminal sequences and 828 cumulative sequence rows across 11 races in
41.17 seconds, versus 2,439 rows and about 114 seconds for exhaustive H=2. All
three trajectories were exact: `ABS → ADD1 → DONE`, `ADD1 → ABS → DONE`, and
child `(0,)::ADD1 → root::ADD2 → DONE`. The optimization is therefore licensed
for the visible Int planner; broader typed factors and public transfer remain
the capability-critical work, not another beam implementation.

The compiler-owned factorized H=1 root planner also reached its first honest
information boundary. Its first two-Int arm scored all 212 compiler-valid
actions through 636 target-free predictions comprising 1,257 unique physical
operator races, then stopped before the
motor on a two-way 4,500 pA value tie: `subtract(SCOPE1,SELECTED)` and
`subtract(SCOPE1,SCOPE0)`. Because the declared public `b` values are all
nonnegative, selected `abs(b)` and scope `b` are observationally identical.
This is `STOP_TYPED_FACTOR_H1_ROOT_VALUE_ALIAS`; the other three arms remained
unrun, and no grid, incumbent, current, or candidate-bank tuning follows. The
missing information is now explicit: execution score defines the leading
physical tier, then generic prompt-derived operand/literal evidence must resolve
only within that tier so lexical evidence can never override worse behavior.

That hierarchy now passes its sole four-task public H=1 run. Complete physical
value tiers were formed before prompt evidence existed; generic prompt binding
then selected the exact tier-local action for both operand orders (`a-b`,
`b-a`) and both comparison orientations (`x<2`, `2<x`). Every task produced
the exact program and outputs through one structural prediction and one real
compiler-owned actuation, with immutable checkpoints. The run took 1.38 s.
Compiler-owned batching made the audited operator path 2.84× faster, while
compiled structural races made the supported 83-race successor path 9.71×
faster with bit-exact ledgers. This is
`PASS_TYPED_FACTOR_H1_PROMPT_HIERARCHICAL_FRESH`, not family-transfer evidence:
the next capability gate varies names, prompt surface, literals, and incumbent
shape before factorized H=2 or any protected milestone is opened.

That transfer gate now passes 4/4 without changing the frozen hierarchy. New
parameter names, paraphrased subtraction prompts, the signed literal `-2`, new
case grids, and unseen binary/nested incumbents produced complete leading tiers
of 2/2/3/2 actions; prompt evidence selected the exact action inside each tier,
and every task closed through one structural prediction and one real actuation
in 3.27 seconds total. `PASS_TYPED_FACTOR_H1_FAMILY_TRANSFER_V1` establishes a
small visible public factor-transfer boundary, not a family-disjoint coding
task result. It licenses the factorized H=2 lift; List, Simple-SWE, M0, and all
protected surfaces remain closed.

The first public factorized H=2 created-child gate then stopped before any
actuation. On the left Int arm there was no H=1 all-case solution, but complete
STOP-bounded expansion admitted all 83 first actions and evaluated 6,414
terminal sequences; every exact leading sequence had structural `UNKNOWN`, so
the right arm remained unrun. A post-run structure-only diagnostic found the
declared first `subtract(SELECTED,SCOPE0)` transition itself uniquely routed to
`UNKNOWN`, making all seven fields at all 40 frozen paths unknown; the child
`(0,)` therefore never became a supported hypothetical site. This is
`STOP_TYPED_FACTOR_HORIZON2_INT_CREATED_CHILD_V1`, not a reason to tune the
fixtures. The next mechanism increment must teach generic factorized structural
routes rather than exact action-pattern rows. Provenance is deliberately
narrow: a concurrent List-terminal-delegate edit changed the planner checkpoint
before this run, so the artifact is not evidence from untouched `410869b`.

The newly named V2 gate spent the exact same two tasks, target declarations,
order, and literal bank against the compiler-generated structure curriculum at
`a31f647`. The structural repair worked: the declared root-subtract then
created-child-absolute trajectory was one of four value-exact, structurally
supported leading sequences. The remaining ambiguity was informational rather than
structural. The prompt observer recognized the words but returned `UNKNOWN`
with `prompt has no complete supported operand relation` for the nested
`subtract(absolute(alpha), beta)` expression; all four sequence currents were
0 pA and the unique 650 pA STOP cell won. Thus
`STOP_TYPED_FACTOR_HORIZON2_INT_CREATED_CHILD_V2` banked after 1,317 sequence
predictions and 8.68 seconds, before actuation, leaving the right arm unrun. No
fixture or mechanism tuning follows. The next generic increment is a
compositional prompt-factor representation for nested unary/binary relations;
the compiler-generated structural successor is no longer this gate's blocker.

The separately named V3 gate spent that smallest generic prompt increment once
against the unchanged V1/V2 protocol. Both public Int arms had no physical H=1
all-case solution. Their nested relations each projected one immutable ordered
sequence checkpoint, selected the declared root subtraction followed by the
absolute edit at its compiler-created child, then completed through exactly one
real edit per replan and terminal `DONE`. Final typed programs and all outputs
matched the frozen declarations, so the banked result is
`PASS_TYPED_FACTOR_HORIZON2_INT_CREATED_CHILD_V3` (2/2 in 145.85 seconds).
This closes the visible created-child mechanism seam and licenses the next
public source-editing increment; it is not List/Bool, Simple-SWE, M0, protected,
C1, family-transfer, or broad coding evidence.

That four-task public source-editing increment's raw V1 artifact reported
`STOP_PUBLIC_FACTORIZED_SOURCE_EDIT_CANARY_V1` (0/1 attempted), but is banked as
`STOP_INVALID_CANARY_GUARD`. On the
first `abs(value)-3` arm, identity and the one-edit source failed their disjoint
sandbox cases, no H=1 exact action existed, and the immutable planner selected
root subtraction before replanning to the absolute edit at child `(0,)`. The
runner stopped before the two-edit phase because its max-edit control counted
that selected-but-unactuated second decision as if it were another completed
edit, then rejected the otherwise valid first-edit prefix. This localizes an
adapter accounting defect rather than a cell-planning failure. The exact canary
was not retried; its raw artifact remains immutable.

A separately named V2 kept the exact four tasks, cases, order, literal bank,
and created-child V3 prerequisite, changing only the target-independent
applied-action extractor to use checkpoint steps that record a real actuation. Both
scalar operand orders and `List.map(abs)` then passed their incumbent, no-H1,
one-edit-bound, exact-program, terminal-`DONE`, disjoint-sandbox, and immutable-
checkpoint checks. The fourth task, `List.map(value-2)`, stopped before any
actuation because prompt evidence did not uniquely resolve its leading
structure tier. The banked result is
`STOP_PUBLIC_FACTORIZED_SOURCE_EDIT_CANARY_V2` (3/4 in 721.31 seconds). This is a
genuine generic nested map-plus-binary prompt-factor boundary, not another
accounting failure. It establishes a small public source-generation bridge but
does not open canonical Simple-SWE, M0, protected data, held-out, or C1.

The fresh two-task recovery is now spent at
`STOP_PUBLIC_FACTORIZED_SOURCE_EDIT_RECOVERY_V1` (1/2 in 416.17 seconds).
`4 - abs(datum)` completed through two real edits, terminal `DONE`, exact source,
and the disjoint sandbox. On List-map-add, the repaired prompt representation
selected and actuated the generic `MAP_BIND` action, producing the exact identity
map, but replanning stopped before the binary body action because the
already-materialized map prompt did not resolve that leading structure tier.
This separates candidate generation from post-map replanning: the former now
works, while the latter needs one generic existing-map-body projection. The
cohort will not be retried or replaced by another two-task microprobe; after that
mechanism repair, the next adjudication is a broad canonical public-TRAIN
Simple-SWE bridge. Protected splits, M0, held-out data, and C1 remain closed.

That broad bridge is now spent at
`STOP_PUBLIC_FACTORIZED_SIMPLE_SWE_TRAIN_BRIDGE_V1`. The frozen shared planner
attempted 7 of 30 round-robin public-TRAIN tasks before the declared family
failure limit fired: scalar-offset passed 0/3, List-map-add passed 0/2, and
List-map-absolute passed 2/2. Every scalar and map-add arm stopped before
actuation with the same `prompt evidence did not uniquely resolve the leading
structure tier` boundary; both absolute-map arms completed
`MAP_BIND → APPLY1 → DONE` and passed the canonical sandbox. The checkpoint
remained immutable, wall time was 2,781.72 seconds, and the artifact SHA-256 is
`f748e50cc98ddc987bcdbbc5804b4203e04b0c0376509d66f374f822781fae32`.
Because the same prompt-tier defect now spans scalar and binary-map families
after the dedicated nested/container repairs, H=2 prompt-projection patching is
retired. The next coding mechanism is **bottom-up typed semantic synthesis**:
compose typed semantic factors from execution-grounded primitives instead of
projecting more natural-language patterns onto pre-enumerated H=2 trajectories.
This remains visible-output public-TRAIN adapter evidence, not prompt-only
completion, protected/held-out, M0, or C1 evidence.

That replacement now passes its first broad gate. One shared frozen bottom-up
synthesizer completed all 30 selected canonical public-TRAIN tasks: 10/10
scalar offsets, 10/10 `List.map(add)`, and 10/10 `List.map(abs)`. It selected
programs from physical semantic-vector races at exact AST costs 3/5/6, and all
960 property cases withheld from selection passed in the canonical sandbox.
The synthesizer saw the typed signature and four regression input/output rows
per task; it did not see the prompt, task/family identity, or the 32 property
rows. One immutable checkpoint produced 17,220 candidate evaluations, 110,200
operator predictions, and 270,200 fitted-GLIF races in 8,937.22 seconds. The
exact artifact is
`experiments/r3_pilot/public_typed_semantic_simple_swe_train_bridge_v1.json`
(SHA-256 `09fbaef134d2ba18a68b513a89d7df5ab8708c8be84940e71a29b710f7ff9767`).
This is `PASS_PUBLIC_TYPED_SEMANTIC_SIMPLE_SWE_TRAIN_BRIDGE_V1`: physical
programming by examples on three TRAIN families, not prompt-only completion,
family-disjoint transfer, M0, protected, held-out, Path-2, or C1 evidence.

The licensed speed pass is now complete without changing that verdict or its
immutable artifact.  The semantic DAG authenticates the frozen operator once
at the enclosing inference boundary instead of re-hashing every fitted array
around every node, while the operator projects one-hot physical events by
column gather and runs bounded multi-row GLIF races.  A fresh replay of the
first round-robin task from each of the three banked TRAIN families preserved
the selected candidate, program/source hashes, candidate/class counts, ordered
operator-race digest, checkpoint identity, and sandbox result exactly.  Their
combined recorded task time fell from 890.21 seconds in the immutable artifact
to 21.05 seconds (42.3x); individual reductions were 263.93→2.33 seconds,
602.40→16.71 seconds, and 23.87→2.01 seconds.  This is a three-task engineering
replay, not a rerun of the frozen 30-task adjudication and not new capability
evidence.  It makes the six-family TRAIN breadth step practical.

Commit `fb83d20` adds a second, narrower runtime cut: exact typed-coincidence
detectors now drive all candidate rows for one target case through one fitted
GLIF population, and semantic-vector races batch by case. Repeated exact
matches, mismatches, four current regimes, semantic decisions, and the frozen
projection/detector/operator/current/tier identities remained byte-exact. A
64-row synthetic physical-phase fixture measured 51.72x in the implementation
run and 73.61x in an independent replay. This is detector-kernel evidence, not
an end-to-end synthesis speed or capability result.

The first generic breadth mechanism layer is now banked without opening any of
those six task outputs.  The live compiler/synthesizer has one retained-frontier
path: deterministic first/lowest-cost physical observational representatives
are carried forward across root cases and every visible bound list item, while
desired outputs remain absent until the exact-vector race at the current tier.
Generic `minimum` and `maximum` compose the existing less-than and copy races;
self-posed `max(n,0)`, symmetric clamp, and mapped-nonnegative programs closed at
AST costs 3, 5, and 6.  The same layer adds bounded Char/String physical output
factors, strict decoding, type-total physical copying, and complete UNKNOWN
inhibition so later String composition cannot win through an unsupported value.
This is commit `75513d2`, a mechanism result only: it is not a six-family TRAIN
gate, prompt-only solving, transfer, DEV/eval, M0, held-out, or C1 evidence.

Commit `00ed3e5` extends that same desired-independent frontier with generic
`sum`, `less-than`, and predicate `count`. A physically retained Bool predicate
at cost 3 licenses its structural binder at cost 4 before `count` may compose at
cost 6. Self-posed checks selected exact `count(3 < item)` for the algebraically
equivalent surface `item + 2 > 5`, and exact cost-9 sum-of-mapped-clamp. The
checkpoint and ordered race ledger were deterministic. This remains mechanism
evidence. A reviewer also found that a diverse four-item, 25-literal frontier
could exceed 90 seconds, so a desired-independent STOP-on-work-overflow bound is
required before the broad TRAIN gate; silent top-k pruning is not licensed.

Commit `3daea4c` completes the generic String mechanism layer.  A
compiler-private `AsciiClass` has a frozen 12-entry catalog spanning broad ASCII
classes plus English vowels and consonants; it cannot enter public callable
signatures.  The physical substrate has 13 class cells including UNKNOWN and
1,141 class-by-printable-Char coincidence cells (12 x 95 plus UNKNOWN).
Generic `contains`, `filter`, and `reverse` compose at exact AST costs 3, 6,
and 2; the retained predicate licenses its Char-to-Bool binder at cost 4, and
`reverse(filter(...))` closes at cost 7.  Contains decisions pass through the
class/Char coincidence and Bool races, while filter and reverse copy String
length and characters through physical races.  Ordered String input shapes are
part of the desired-independent retention key.

The catalog-off control preserves the catalog-dependent compiler inventory,
node order, evaluation contexts, and physical class/contains schedule through
the intervention, but drives every class terminal to physical UNKNOWN so no
filter representative can win.  Downstream output-copy work is allowed to stop
causally after UNKNOWN rather than replaying treatment decisions.  Compiler
whitespace semantics cover all six ASCII whitespace characters, while
non-printable or over-26-character synthesis inputs remain UNKNOWN.  Self-posed
telemetry was 63 predictions/74 races for direct reverse, 364/894 for vowel
filter, 412/996 for decimal-digit filter, and 484/1,274 for reversed consonant
filter; the direct-reverse ordered race digest is
`4ed8ab1c38ec4350a25314a0581c2bd5b5cdde2a1b8503a67b5154fe83a0dace`.
The operator checkpoint is
`92944516be48275ba7633a29a298707393010b85c3a0c60be4b6c73668d7e76e`,
the enabled semantic checkpoint is
`962c2359fb41d7444a24bcacdeb2679caa3d5406719e6a6c85887c14a174598e`,
and the catalog-off semantic checkpoint is
`8931b4e106e500fa78c93001fe153c740660effa2934193e55668fc1cbedc612`.
Root verification passed 95 tests.  This is self-posed generic mechanism and
control evidence only, not the broad TRAIN breadth gate or transfer/capability
evidence.

Commit `7916741` completes the desired-independent semantic-frontier work
bound without changing the public synthesis interface or opening any of the
six unspent TRAIN-family desired vectors.  It admits complete exact-cost tiers
under four fixed private limits: 256 retained representatives per explicit
typed context bucket, 6,000 declared root candidates, 30,000 physical operator
predictions, and 80,000 physical races.  Overflow returns
`STOP_SEMANTIC_FRONTIER_WORK_BOUND`; it never silently drops a representative,
and an interrupted tier never reaches the desired-output race.  Runtime
admission and checkpoint hashing consume the same canonical aggregate-plus-
String schema table, including the 29-race worst-case String copy.

The originally proposed 128-representative cap was transparently superseded
before implementation and before any gated spend.  The unchanged cost-6 count
and scalar exhaustive-STOP mechanisms reached 171 bound-Int and 168 root-Int
representatives, falsifying 128.  Raising the fixed cap to the next power of
two, 256, restores approximately the original headroom principle: 171→256 is
50%, versus the original sentinel calibration of 87→128 at 47%.  A tighter
192 cap and the final 256 cap produced the same prediction-first Sentinel-B
prefix, so no task outcome selected the correction.

The duplicate-input 25-literal Sentinel A remains an ordinary exhaustive STOP:
4,797 declared roots, 4,757 supported roots, 87 root semantic classes, 23,642
predictions, 63,456 races, and normal race SHA-256
`2ca6e201b448b8b74070c485fe72ec66502fc303fc66e8f3f33f05c61fa82ff2`.
The diverse permutation Sentinel B stops before prediction 30,001 at cost 3,
with 30,000 completed predictions, 77,964 races, 1,651 declared roots, 1,441
supported roots, 174 retained representatives, no cost-3 desired race, and
completed-prefix SHA-256
`506b110fac0558063293d08527c8c0ae6ad89849e5d97b24859d6bd82123970c`.
Changing its inconsistent desired vector preserves the complete prefix.  The
new bounded semantic checkpoints are enabled String
`9ac87703dae72cf6efeee52b5ceaa953dffc072dc5303f3348d8a8b6f51486b6`,
catalog-off String
`ed7dff6036ab522840bca1c0a1b6fd7c2e0de0768be813bb33b1920b5f291af3`,
25-literal sentinels
`13948fd329c79488f2e6532f3e013a10864db584db1c35b4219a1abe3fc9c477`,
count
`4f0ee145047bccba72745a65afd9c261f7dd6d7a845ecc16c90fe104ff2e0ad6`,
and clamp-map-sum
`09c1e11ba835fd229b2741b0816de4704ea99ae46880ebbc5247e80f862979a2`.
An independent read-only audit passed; implementer and root runs each passed
100/100 tests in 103.36 and 98.44 seconds.  This is a self-posed runtime-safety
mechanism only, not broad TRAIN, prompt-only, transfer, DEV/eval, M0, held-out,
Path-2, C1, SWE-bench, Gemma, or Opus capability evidence.

The compact typed-output prerequisite is now banked.  Commit `82b57a0`
replaced every production 3,284-cell dense value with an immutable 104-byte
winner-plus-digest encoding, cut the active operator, semantic, vector-race,
prompt-literal, and fitted-GLIF paths over to the compact interface, and left
no dense compatibility route or production audit materialization.  The sole
canonical capture is
`experiments/r3_pilot/compact_typed_output_compact_v1.json` (raw SHA-256
`f503820711ebc597aeac6b96e6d89dce17c48be7b756b182619a7238b531e179`,
payload SHA-256
`beab356461d443fc5ac5f5cb5198e755e3e6927b48a8aa860bc8fdbd68b3604e`).
All 44 frozen gates passed with exact scientific golden
`287987cea7306778d79ba0e726b50fde304860e00f8ae8b2d2057b0c432920ee`:
2,000 Strings peaked at 705,722 bytes (75.75x below the recorded dense
baseline), 1,000 detectors at 592,795 bytes (44.89x), the String-heavy frontier
at 1.35%, all nine runtime result hashes remained exact, and both work-bound
sentinels preserved their STOP point and race digest.  The artifact used zero
external task/corpus inputs.  This is representation, exact-physics, and
engineering-performance evidence only; the six-family TRAIN outputs remain
unspent and every DEV/eval, M0, held-out, Path-2, and C1 claim remains closed.

The prompt-language continuation has also banked its checkpoint-free input
substrate at `c89a3c6`.  It authenticates the frozen callable frame, replaces
declared names by entry/argument roles, represents signed integer lexemes as
`<INT>` plus a separate physical copy record, and drives every ordered 1--4
token span through a deterministic 12-of-512 fitted-cell assembly.  It has no
teacher, factor classifier, desired output, family/task identity, candidate
program, or trained state.  The learned span-to-semantic-factor mapping and its
preregistered TRAIN capacity split remain unbuilt; DEV therefore remains sealed.

The overfit-safe continuation is fixed in three layers. First, improve runtime
and extend only on the six unspent TRAIN families using generic typed
combinators rather than family operators. Second, learn and freeze a
prompt-span-to-semantic-factor bridge on a preregistered TRAIN capacity split;
do not revive static whole-prompt projection. Third, after both mechanisms are
frozen, spend the 27 family-disjoint public DEV tasks once with paired
visible-I/O and canonical prompt-only arms. Opening DEV visible outputs before
the prompt bridge is frozen would leak the transfer surface into its design.

On the Path-2 reading mainline, the first calibration-v3 TRAIN mismatch is now
localized rather than guessed. At Genesis step 7,933, eleven segments grown at
step 249 under a different winner context shared 8/11 cells with the current
context and therefore reached theta-match 16, but not theta-active 18. Their
owners were outside the observed token columns, so the current novel key still
allocated all 11 owners with no prediction or punishment. The exact-fanout
scorer had conflated global candidate/matching diagnostics with exact-key
recognition. A one-run `(18,20)` counterfactual pushed those same rows below
matching while preserving the local outcome, but it cannot select that arm:
the existing q=255 production-overlap control already stops `(18,20)` through
stored-owner drift. A later scorer then made two different category errors.
Its first public-TRAIN probe treated same-token active-path surface prediction
as owner capture even though native LEARN allocated a fresh exact-source tuple.
Its corrected replay verified 975,952 monolithically allocated novel keys and
69 exact complete-tuple repeats before assuming that the next novel full key
also had to allocate all 11 rows. At the 2 Chronicles boundary, however, the
exact repeated `49,1077` source had already learned successor 1260, and 1260
shares production column 4078 with novel successor 1073. Native persistent
identity is factorized by output column, so this topology predicts reuse of the
shared exact-source owner and allocation in the other ten columns rather than
eleven fresh rows.

A target-independent native fixture now pins that contract with two successor
codes sharing exactly one column: first allocation is 10 with one predicted
column and only the shared owner reused; the complete tuple differs; replay is
0 allocation with all 11 columns predicted; terminal storage is 21 rows. This
reclassifies the spent boundary as
`STOP_INVALID_MONOLITHIC_ALLOCATION_CRITERION`, not a Path-2 failure. The raw
probe, marker, and artifact remain immutable, and the masked full-state row and
remaining TRAIN behavior remain unknown. Any successor scorer must preserve
both matching-versus-active telemetry and per-column identity factorization,
requiring stability of the complete tuple while summing actual allocations
rather than enforcing `11 * novel_keys`. This is not a calibration-v3 retry,
threshold change, or license to open protected/C1 data.

The dense Path-2 TRAIN build has independently reached `BUILD_COMPLETE` for all
49 books and 2,117,412 tokens, with 23,286,405 actual segments and a 1.490 GB
durable checkpoint. The final isolated Genesis LEARN/READ_ONLY measurement has
not run, so this is resumable infrastructure state only: there is no cap PASS/
STOP, C1, held-out, or capability verdict yet.

A separate target-independent factorized scorer now pins the successor
identity contract in Python plus the actual native one-shared-column fixture.
It requires stable complete `(source tuple, observed token)` keys, immutable
token-column codes, owner stability for every exact-source column factor,
globally non-aliased owner columns, zero-allocation exact replay, and terminal
storage equal to the sum of actual allocations. Matching and active segment
telemetry are recorded separately and never gate identity. This is mechanism
evidence only. The next licensed adjudication is a newly named, source-bound,
public-TRAIN replay from empty state; it is not a calibration-v3 retry and does
not open held-out or C1 data.

That adjudication is now prepared but remains unspent. Its source-bound runner
starts both native and scorer state empty, consumes only `VerifiedTrainInput`,
and checkpoints at book boundaries under an exclusive spend marker and atomic
manifest. A 62-byte record per token makes the complete scorer journal at most
131,282,288 bytes without a durable Python `(source,column)` map; one active
native checkpoint bounds final durable state at 1,621,955,424 bytes and
publication peak at 3,120,208,302 bytes. Resume reconstructs the scorer in
O(authenticated prefix tokens), with an explicitly unmeasured 3–8 GiB live
Python-memory estimate. The real 49-book preflight reports
`PREFLIGHT_READY_NOT_RUN`, zero held-out/C1 materialization, and no marker,
manifest, checkpoint, or corpus execution. The old 1.490 GB completed dense
build is not imported: the scorer needs the chronological native outcomes of a
fresh replay.

### Phase R5 — Scale to billions (continuous)

Tiered runtime per capacity envelope: CUDA ≤200k / CPU frontier to 6M /
procedural sparse ≥1e7 configured, ~1.9e8 residents max on this box.
Event-proportional work semantics (adapter L3) is the prerequisite —
current semantics are untrainable at 1M residents (0.09 tok/s); with
event-proportional updates the measured estimate is 160–750 tok/s
independent of configured N. Multi-GPU/multi-box distribution is the
post-1e8 discussion, deliberately out of scope until single-box learning
is proven.

## Immediate queue

1. **COMPLETE:** bank the semantic 30/30 artifact and make its 8,937-second
   physical path materially faster without changing winners, race ledgers, or
   checkpoint identity.  A parity-checked first-three-task replay is 42.3x
   faster in aggregate.
2. **COMPLETE:** extend the generic semantic basis with retained Int extrema,
   the Char/String output substrate, List aggregates, String combinators, and
   the private desired-independent frontier work bound.  No six-family desired
   vector was opened to implement or calibrate the bound.
3. **IN PROGRESS:** the preregistered compact authenticated encoding is banked
   with all 44 gates passing.  Next freeze and spend the broad six-family
   public-TRAIN visible-I/O gate once.
4. Build the TRAIN-only prompt-span → semantic-factor bridge and freeze it
   before any family-disjoint desired vector is observed.  The checkpoint-free
   physical span input is banked; the learned factor mapping remains.
5. Spend one paired 27-task public DEV gate only after the visible-I/O engine
   and prompt-only bridge are both frozen. M0 and canonical eval stay sealed.
6. Spend/resume the prepared source-bound Path-2 factorized scorer replay from
   empty state in bounded public-TRAIN chunks; do not import the completed
   dense build or open C1/held-out material.
