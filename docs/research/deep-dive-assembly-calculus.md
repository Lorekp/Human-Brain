# Deep dive: Assembly Calculus / NEMO — does their Hebbian language acquisition escape the E3 static-prior trap?

**Date:** 2026-08-25
**Status:** primary-source mechanism comparison; integration verdict
**Sources verified directly:** `arXiv:2507.11788` (Mitropolsky & Papadimitriou,
*Simulated Language Acquisition in a Biologically Realistic Model of the Brain*,
full text extracted from PDF); `github.com/dmitropolsky/assemblies` cloned and read
(`brain.py`, `brain_util.py`, `parser.py`, `recursive_parser.py`,
`word_order_int.py`). PNAS 2020 (`pnas.2001893117`) page fetch was denied by the
operator mid-session, so PNAS-level claims below are carried via the NEMO
paper's own description and reference [32] and are labeled as such.
Our side: `t-route-preregistration` (`trained-inter-space-preregistration.md`),
`t-route-e1e2-verdict.md`, `t-route-program-conclusion.md`.

## Bottom line

**No — for our task class, NEMO does not escape the E3 static-prior trap; it
avoids entering it.** Their results are real but live inside a regime defined by
three ingredients that Bible-QA verbatim-quote routing lacks:

1. **Repetition-rich, closed-loop exposure** (~10 sentences per word × τ ≥ 2
   rounds each, with the same k-cap competition active during learning), which
   lets multiplicative Hebbian growth `(1+β)^t` compound until learned weights
   dwarf the random prior. The prior seeds *which* neurons get recruited; it
   does not decide the final winner — but only because exposures repeat.
2. **A hand-designed architecture that carries compositionality**: distinct
   areas per role (LEX1 nouns / LEX2 verbs, ROLE agent/action/patient,
   SUBJ/VERB/OBJ, MOOD), hand-picked fibers, several with elevated `p`/`β`
   baked in, mutual inhibition among ROLE areas, and externally driven
   grounding (VISUAL/MOTOR/context assemblies co-fired by the experimenter).
   Word order is learned as plasticity between ROLE and SYNTAX areas —
   **position lives in *which area fires when*, never inside token codes.**
   This *confirms* our E3 statement (position is not expressible in a
   token-identity substrate) rather than refuting it.
3. **Small, well-separated item sets**: tens of words, one assembly each, with
   stability-vs-instability as the success criterion. Their own theory line
   (classification result, COLT 2022, ref [7]) guarantees separation-based
   regimes; the published sequence-memory limit is **20–40 assemblies**
   (Dabagia et al., Neural Computation 2024, ref [6], cited in NEMO §2.6).
   A KJV verse is 20–60 tokens; who-said routing must discriminate among
   hundreds of heavily overlapping quotes. Both sit at or far beyond their
   demonstrated envelope.

Their mechanism differs from our failed accumulator in ways that matter, and
some are portable (see adopt/adapt/avoid). But the program-grade E3 conclusion —
fixed connectivity + local rules + token-identity SDRs cannot express
position-addressing — applies to NEMO *a fortiori*, because NEMO never puts
position into token codes: it spends architecture on it.

---

## Q3 first (foundation): the actual update rule, from the code

Everything below is read off `brain.py` in `dmitropolsky/assemblies` (line
numbers from the cloned HEAD) unless attributed to the paper.

**State.** Areas with `n` neurons (10⁵–10⁶ typical), `k` winners per round
(k ≈ √n; experiments: k_LEX=50, k_CONTEXT=20, others 100), plasticity β ≈
0.05–0.06, connectivity p ≈ 0.01–0.05, initial weights 1 on Bernoulli(p)
synapses. Fibers (area→area) exist only where hand-declared.

**Discrete synchronous rounds.** One call to `project()` (brain.py:272) is one
round: every target area computes its next winner set from the *previous*
round's winners, and all areas commit `_new_winners` simultaneously
(brain.py:306–308). No continuous time, no membrane state anywhere.

**Selection — the k-cap.** `project_into()` sums incoming weights from firing
sources (`prev_winner_inputs`, brain.py:341–348) and takes the **k largest**
(`heapq.nlargest`, brain.py:430). Local inhibition is entirely abstracted to
this top-k; ties arbitrary. Inter-area inhibition is a separate, rarer device:
among a mutually-inhibited set of areas only the one with greatest total input
fires (used *only* among the three ROLE areas in the language system, paper
§2.3).

**Plasticity — multiplicative, winner-only, unbounded.** After selection, every
synapse from a presynaptic winner onto a postsynaptic winner is multiplied:
`connectome[j, i] *= 1 + beta` (stimulus→area: brain.py:499–500; area→area:
brain.py:542–544). There is **no LTD term, no decay, no pruning, no weight cap,
no homeostat anywhere in the file** (grep confirms). Weights are therefore
`co-firing count` encoded as `(1+β)^t`. Per-fiber β is settable
(`update_plasticity`) and a global `disable_plasticity` flag exists.

**No fan-out cap.** Any winning presynaptic neuron potentiates onto *all*
winning postsynaptic partners it connects to. The only sparsity controls are
structural: k winners per round, Bernoulli(p) support. New synapses appear only
onto first-time winners, freshly sampled Bernoulli(p) (brain.py:528–538,
559–561).

**Lazy simulation.** Non-explicit areas never simulate all n neurons: they track
only the ever-fired support `w`, sample k hypothetical newcomers' input counts
from a truncated-normal approximation of Binomial(total_inputs, p) above the
`(n−k)/n` quantile (brain.py:382–414), and synthesize provenance for
first-time winners by sampling (brain.py:462–474). This is why they reach 10⁵–
10⁶ neurons on a laptop — a trick directly relevant to our EP/staged-membership
seams.

**What is *not* learned.** Explicit input areas cheat constructively: PHON and
MOOD are initialized by `activate(area, index)`, which sets winners to the
contiguous index block `[k·index, k·index+k)` and freezes them
(brain.py:265–270, `fix_assembly`; `fixed_assembly` short-circuits plasticity
and competition entirely, brain.py:332–337). Word forms, moods, and semantic
representations (VISUAL/MOTOR/context assemblies) are experimenter-given.
The earlier parser goes further: `recursive_parser.py` drives comprehension with
per-word hand-written PRE_RULES/POST_RULES scripts that inhibit/disinhibit
specific fibers (e.g. generic_noun, recursive_parser.py:55–100) — the grammar is
*programmed into the protocol*. Only the 2025 NEMO paper learns anything
syntactic, and only constituent order over S/V/O chains; the paper concedes the
MOOD assemblies are its one constructively-planted exception (Discussion).

**Learned operations.** At the calculus level (PNAS 2020 framing, carried here
via NEMO refs [32],[26]): the primitive is **projection** — firing assembly A in
area X into fresh area Y creates a copy Ā stabilized by plasticity;
reciprocal projection, association (co-fire two assemblies → bind them),
merge, and pattern completion are protocols built from repeated projections.
This repo implements exactly the projection primitive plus protocol scripting.

---

## Q1: How does this differ from raw outer-product accumulation?

Our T-ROUTE estimator (frozen preregistration §1–2): sparse random-init W
(fan-in 32, U(0.5,1.5)), **additive** one-shot accumulation
`dW[sdr(tq) × sdr(t_p[j′])] += M` over a single pass of 200 episodes, then a
*frozen* readout `drive = W @ q_pop`, top-K=128. Side-by-side:

| Dimension | T-ROUTE accumulator (E1–E3) | Assembly Calculus / NEMO |
| --- | --- | --- |
| Update sign | Additive `+= 1` per event | Multiplicative `*= (1+β)` per co-firing |
| Time structure | One-shot batch, then frozen | Discrete rounds; learning *is* the running dynamics |
| Competition during learning | None (union states accumulate passively) | k-cap selects winners **every round**; only winners potentiate |
| Recurrence | None (feedforward Q→P map) | Self-projection + bidirectional fibers; assemblies are attractors |
| What plasticity acts on | All anchored token pairs, weighted by frequency | Only the current winner set — rich-get-richer lock-in |
| Prior's role | Decides winners outright (static ~170-bit common-byte pool) | Seeds initial recruitment; final winners sculpted by compounded growth |
| Position/role | Must be expressed inside token-identity SDRs | Allocated to dedicated areas + firing order + inter-area inhibition |
| Supervision | None (corpus statistics only) | Grounding: meaning assemblies co-fired by the experimenter |
| Scale of learned store | 200 passages, 260 QA items | ℓ ≈ tens of words, one assembly each |

**Where compositional structure actually comes from.** Not from the plasticity
rule alone — the rule is nearly the simplest Hebbian rule there is. Structure
comes from the interaction of (a) k-cap competition closing the loop between
selection and potentiation, (b) recurrence turning frequently co-fired sets into
self-sustaining attractors (Property 1 "stability" is literally this), and
(c) an area alphabet whose wiring diagram encodes the task grammar. Association
of two assemblies, sequential chaining, and role binding are all implemented as
plasticity between *concurrently or consecutively firing* populations in
*different areas* — composition is spatial (which areas) and temporal (which
order), never vectorial (within-code positions).

## Q2: Is their language task close enough to Bible-QA to inherit the verdict?

**Their task**, decomposed: (1) pair each pre-given phonological assembly with
pre-given semantic assemblies under massive repetition — supervised associative
pairing; (2) let differential pathway strength (LEX1↔VISUAL elevated, LEX2↔MOTOR
elevated) sort nouns from verbs, measured as stable-attractor-here vs
wobbly-set-there; (3) learn a 3-step order chain (MOOD→ROLE→SYNTAX) from
repeatedly presented, externally sequenced sentences, with thematic roles
**pre-computed and pre-placed in ROLE areas by the experimenter** (§2.3: the
scene "prepares" ROLE assemblies connected to the right lexical assemblies);
(4) generation = replay the learned chain via trigger-driven disinhibition.

**Bible-QA who-said** decomposed: store verse streams from text alone (no
grounding co-presentation, no role annotation, single exposure per verse); at
query time match a verbatim-quote cue against stored spans sharing most
vocabulary, locate the span, and read off its adjacent attribution.

Three disanalogies decide it:

- **Exposure regime.** Their compounding needs ~10+ co-present repetitions per
  item; ours gets one occurrence per verse and the query is never co-present
  with the answer. E2 already measured what one-shot additive accumulation
  yields (query-independent winners); switching to `*=(1+β)` cannot conjure
  repetitions that do not happen — it only accelerates lock-in *when* events
  recur.
- **Item separability.** Their discriminative successes (noun vs verb; correct
  vs wrong area) are binary and between *well-separated* assemblies. Who-said is
  many-way discrimination between near-duplicate spans ("And the LORD said…"
  recurs hundreds of times) — exactly the regime their separation-based theory
  does not cover and their experiments never probe.
- **Capacity and sequence limits.** One LEX-class area at n=10⁵, k=50 holds on
  the order of 10³ low-overlap assemblies; the corpus is ~3×10⁴ verses. The
  published replay limit of 20–40 chained assemblies (ref [6]) brackets
  verse-length sequences from below. Scaling to corpus-size episodic storage in
  fixed random connectivity is not demonstrated anywhere in this literature and
  would resurface our interference/static-prior problem in a new place.

**Could their mechanism do who-said routing anyway?** The nearest-fit variant —
bind quote-assembly ↔ speaker-assembly by repeated co-firing, retrieve by
pattern completion — is a pairwise associative memory, the one task shape where
local Hebbian genuinely shines. But it presupposes co-present, repeated,
segmented quote/speaker pairs, i.e. it reformulates the task into the shape
their regime supports (our path 3 "task pivot"). Under our constraint set
(single-pass corpus encoding, query-time routing, identity-bit substrate), their
own machinery would land where ours did: the winner set is decided by whatever
the static prior favors whenever the plastic signal is sparse relative to prior
mass and items are not separated. **The E3 negative transfers; their papers
provide no counterexample because they never run this regime.**

## Q4: Integration verdict

The E3 program-grade statement survives contact with NEMO: nothing in their
system addresses position inside token-identity codes, because they never try —
they buy composition with architecture (areas/fibers/inhibition/timing), which
is a cousin of our path-2 "trained-by-construction connectivity": the prior is
relocated from synapse values into wiring and protocol. What transfers is a set
of mechanisms worth testing *inside* our constraint relaxation options, not an
escape hatch.

### Adopt / adapt / avoid

| Verdict | Mechanism | Why | Where it lands in our decision space |
| --- | --- | --- | --- |
| **Adopt** | k-cap (exact top-k, every round, ties deterministic) | We already read out with top-K once; making selection round-wise *during encoding* converts passive union state into competitive state, so eligibility attaches only to cells that actually won | Compatible with current stack; candidate arm for any post-E3 encoding redesign |
| **Adopt** | Lazy-support simulation (track ever-fired support; sample newcomers above the (n−k)/n quantile) | Same trick family as our EP/bitmap seams; independently validates staged-membership engineering at 10⁵–10⁶ scale | Runtime engineering notes |
| **Adapt** | Multiplicative `(1+β)` potentiation replacing `+= 1` | Enables attractor lock-in under repetition; strictly stronger rich-get-richer than additive | Only meaningful paired with repetition-bearing episodes (synthetic-episode synthesizer already produces 2000+) |
| **Adapt** | Position-as-place: dedicate populations to roles (speaker-area, quote-span-area, attribution-slot) so binding is cross-area co-activation | Directly answers how to express "where" without token-position bits; converges with R3 episodic-store reasoning | Path 2 (trained-by-construction connectivity) design input |
| **Adapt** | Inter-area mutual inhibition for serial role arbitration | Clean primitive for "one speaker candidate wins per step" in QA readout; cheaper than global WTA hacks | Readout/arbitration layer |
| **Avoid** | Expecting tabula-rasa random connectivity + local Hebbian to scale to corpus-scale episodic QA | No evidence anywhere in the program; capacity and separation math point the other way | Reinforces E3 termination rather than reopening it |
| **Avoid** | Counting their "biological plausibility" as unsupervised learning | Grounding, planted assemblies, hand-written fiber scripts, and index-block words are teacher signals; adopting the mechanism means accepting those teachers | Guards against importing hidden supervision unknowingly |

### Cheap falsification of this assessment (if anyone wants to spend it)

A single preregistered probe could refute "the trap transfers": rerun the E2
config with (i) `*=(1+β)` updates and (ii) k-cap winner selection active during
encoding, on the synthetic-episode synthesizer with ≥10 repetitions per
quote/speaker pair, gating on shuffled-question enrichment. Prediction from this
analysis: enrichment recovers only on the repeated-pair stratum and collapses
on held-out single-exposure items — i.e., NEMO's regime works exactly where it
always worked (repeated, co-present, separated) and nowhere else. If
single-exposure VAL items route query-dependently, this document is wrong and
NEMO deserves an installation spike.

### Verification ledger

- Update-rule details: `assemblies/brain.py` lines 272–308 (rounds), 341–348 +
  430 (input sum + top-k), 499–500 + 542–544 (multiplicative plasticity),
  382–414 (lazy newcomer sampling), 265–270 + 332–337 (index-block activation,
  frozen assemblies). No decay/LTD/cap: grep across `brain.py` negative.
- Hand-wired parser protocol: `recursive_parser.py` 50–120 (per-word
  inhibit/disinhibit rule tables); helper-area syntax learner:
  `word_order_int.py`.
- Architecture, grounding, properties 1–3, results (~10 sentences/word, linear
  scaling), generation algorithm, ROLE mutual inhibition, O-initial orders
  harder + provably unlearnable under <50% transitive input, 20–40 sequence
  limit, MOOD-assembly concession: `arXiv:2507.11788` §§2.1–2.6, Discussion
  (extracted full text, `.tmp-research/nemo.txt` during research; scratch since
  removed).
- Our side: frozen estimator and arms (`trained-inter-space-preregistration.md`
  §§1–2), E1/E2 numbers (`t-route-e1e2-verdict.md`), E3 random-span control and
  termination logic (`t-route-program-conclusion.md`).
- Not verified directly: PNAS 2020 operation definitions and proofs (page fetch
  denied); carried via NEMO refs [32],[6],[7],[26] and labeled above.
