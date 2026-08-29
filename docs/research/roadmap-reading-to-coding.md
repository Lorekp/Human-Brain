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

### Phase R0-d — Custom architecture pivot — CURRENT

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

### Phase R5 — Scale to billions (continuous)

Tiered runtime per capacity envelope: CUDA ≤200k / CPU frontier to 6M /
procedural sparse ≥1e7 configured, ~1.9e8 residents max on this box.
Event-proportional work semantics (adapter L3) is the prerequisite —
current semantics are untrainable at 1M residents (0.09 tok/s); with
event-proportional updates the measured estimate is 160–750 tok/s
independent of configured N. Multi-GPU/multi-box distribution is the
post-1e8 discussion, deliberately out of scope until single-box learning
is proven.

## Immediate queue (this week)

1. GPU agent: 1M stability-pass fix + commit (in flight).
2. Multi-timescale trace protocol doc + red tests (R0 keystone).
3. Bible-QA first real run at 128k–1M (R2 start) once R1 lands.
4. Procedural adapter L1/L2 spike (token injection + readout mirror).
5. simple-SWE corpus audit for R4 readiness (unopened stack, verified
   earlier: freeze/reveal machinery 117/0 green).
