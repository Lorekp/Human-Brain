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

### Phase R0-b/race — Decision layer (current)

Per the preregistered fork: if features cannot separate contexts by any
spike-derived family, the degeneracy lives in how decisions form, not how
state is encoded. Candidates ranked by the strategy report (dispatched):
winner-feedback broadcast (HOT-2 seed), sparse associative memory over
assemblies (hippocampal-style hybrid), learned output map over frozen cell
dynamics (reservoir-computing orthodoxy), dendritic-state proxies. The
constant-output signature ([variable-0] modal everywhere) is the target:
any mechanism that makes the race see different evidence for different
contexts destroys it.

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
